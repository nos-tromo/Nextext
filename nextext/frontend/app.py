"""Streamlit UI for the Nextext audio analysis workflow.

The UI is a thin HTTP client over the FastAPI backend defined in
``nextext.api``. All heavy ML work happens server-side; this module
intentionally avoids importing ``nextext.pipeline`` or any module that
chains to it, so the Streamlit Docker image can ship with only the
``frontend`` dependency group installed.
"""

from __future__ import annotations

import io
import os
import re
import sys
import uuid
import zipfile
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import altair as alt
import pandas as pd
import streamlit as st
from loguru import logger
from streamlit.web import cli as st_cli

from nextext.frontend.client import BackendClient, StageEvent
from nextext.frontend.state import (
    check_batch_within_limit,
    default_target_language,
    download_file_name,
    named_entities_list_to_dataframe,
    progress_value,
    result_file_names,
    select_result,
    transcript_list_to_dataframe,
    word_counts_list_to_dataframe,
)

PIPELINE_STAGE_LABELS: tuple[str, ...] = (
    "Transcribing",
    "Translating",
    "Running word-level analysis",
    "Summarizing",
    "Detecting hate speech",
)

OWNER_PARAM = "owner"
_UUID4_HEX_RE = re.compile(r"^[0-9a-f]{32}$")

_DEFAULT_MAX_BATCH_MB = 2048


def _max_batch_bytes() -> int:
    """Resolve the in-memory batch cap from ``NEXTEXT_MAX_BATCH_MB``.

    Streamlit holds the whole upload selection in memory at once, so this
    caps the combined batch size the UI accepts before submitting. Defaults
    to ``_DEFAULT_MAX_BATCH_MB`` MiB; a non-integer value falls back to it.

    Returns:
        int: The cap in bytes.
    """
    raw = os.getenv("NEXTEXT_MAX_BATCH_MB", str(_DEFAULT_MAX_BATCH_MB)).strip()
    try:
        mb = int(raw)
    except ValueError:
        mb = _DEFAULT_MAX_BATCH_MB
    return max(mb, 1) * (1 << 20)


def _ensure_owner_id() -> str:
    """Resolve the per-browser owner identifier, persisting it in the URL.

    The URL query parameter ``?owner=<uuid>`` is the canonical carrier
    for the per-browser identity. On first visit (no param present) the
    function mints a fresh UUID4 hex and writes it into ``st.query_params``
    so Streamlit updates the address bar; subsequent reloads, browser
    history navigations, and bookmarks all re-supply the value via the
    URL without any client-side state.

    This approach was chosen over a ``localStorage`` shim because
    ``streamlit.components.v1.html`` iframes are sandboxed without
    ``allow-top-navigation``: a JS bootstrap attempting
    ``window.parent.location.replace(...)`` is silently blocked, which
    leaves the page stuck on the "Initializing session…" gate forever.

    Returns:
        str: The validated UUID4 hex identifier for the current browser.
    """
    owner_param = st.query_params.get(OWNER_PARAM)
    if isinstance(owner_param, str) and _UUID4_HEX_RE.match(owner_param):
        st.session_state["owner_id"] = owner_param
        return owner_param

    new_id = st.session_state.get("owner_id")
    if not (isinstance(new_id, str) and _UUID4_HEX_RE.match(new_id)):
        new_id = uuid.uuid4().hex
        st.session_state["owner_id"] = new_id
    # Persist into the URL so a reload (or a returning visit via the
    # browser's history) keeps the same identity. Setting an entry on
    # ``st.query_params`` updates the URL bar without an extra rerun.
    st.query_params[OWNER_PARAM] = new_id
    return new_id


@st.cache_resource(show_spinner=False, scope="session")
def _get_client(owner_id: str) -> BackendClient:
    """Return a session-cached :class:`BackendClient` for one owner.

    The cache is session-scoped *and* keyed on the owner_id so two
    browser tabs that happen to land on the same Streamlit session
    cannot accidentally share a client (and therefore an identity).

    Args:
        owner_id: Stable per-browser id carried in the page URL (``?owner=``).

    Returns:
        BackendClient: A long-lived client with the trusted identity
            header pre-bound.
    """
    return BackendClient(owner_id=owner_id)


def _get_owner_id() -> str:
    """Return the owner_id stashed by :func:`_ensure_owner_id`.

    Returns:
        str: The current owner_id.

    Raises:
        RuntimeError: If called before ``_ensure_owner_id`` populated
            the session state — indicates a Streamlit script-flow bug.
    """
    owner_id = st.session_state.get("owner_id")
    if not isinstance(owner_id, str):
        raise RuntimeError("owner_id is not in session state; _ensure_owner_id must run before any backend call.")
    return owner_id


@st.cache_data(show_spinner=False, ttl=300)
def _fetch_languages() -> dict[str, list[dict[str, str]]]:
    """Fetch the language mappings once per session.

    Returns:
        dict[str, list[dict[str, str]]]: ``{"whisper": [...], "target": [...]}``.
    """
    client = _get_client(_get_owner_id())
    return client.get_languages()


def _split_language_lists(
    entries: Sequence[dict[str, str]],
) -> tuple[dict[str, str], list[str]]:
    """Materialize the code->name mapping and the sorted name list.

    Args:
        entries: ``[{"code": ..., "name": ...}, ...]``.

    Returns:
        tuple[dict[str, str], list[str]]: ``(code_to_name, sorted_names)``.
    """
    mapping = {entry["code"]: entry["name"] for entry in entries}
    names = sorted(mapping.values())
    return mapping, names


def _name_to_code(mapping: dict[str, str]) -> dict[str, str]:
    """Return the inverse of ``mapping``.

    Args:
        mapping: Code-to-name mapping.

    Returns:
        dict[str, str]: Name-to-code mapping.
    """
    return {v: k for k, v in mapping.items()}


def _normalize_snapshot_result(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Reshape an API snapshot's ``result`` block into the legacy UI dict.

    The legacy UI consumed Pandas DataFrames for transcripts, word counts,
    and named entities. The API returns the same data as JSON lists; this
    helper rebuilds DataFrames so the existing tab-rendering code stays
    untouched.

    Args:
        snapshot: ``GET /jobs/{id}`` response body.

    Returns:
        dict[str, Any]: Result payload shaped for the UI.
    """
    api_result = snapshot.get("result") or {}
    return {
        "file_name": snapshot.get("file_name"),
        "job_id": snapshot.get("job_id"),
        "source_file_hash": snapshot.get("source_file_hash"),
        "task": api_result.get("task", "transcribe"),
        "transcript": transcript_list_to_dataframe(api_result.get("transcript") or []),
        "transcript_language": api_result.get("transcript_language"),
        "resolved_src_lang": api_result.get("resolved_src_lang"),
        "summary": api_result.get("summary"),
        "word_counts": word_counts_list_to_dataframe(api_result.get("word_counts")),
        "named_entities": named_entities_list_to_dataframe(api_result.get("named_entities")),
        "wordcloud_url": api_result.get("wordcloud_url"),
        "hate_speech_findings": api_result.get("hate_speech_findings"),
        "skipped": bool(api_result.get("skipped", False)),
        "skip_reason": api_result.get("skip_reason"),
    }


def _render_event_delta(container: Any, event: StageEvent) -> None:
    """Surface inline progress hints derived from SSE ``result_delta`` payloads.

    Args:
        container: An ``st.status()`` container to write into.
        event: The SSE event to render.
    """
    delta = event.data.get("result_delta") or {}
    if event.name == "stage_completed":
        if "transcript_segments" in delta:
            count = delta["transcript_segments"]
            if count:
                container.write(f"**Transcript:** {count} segments")
            else:
                container.warning("Transcript empty — file skipped.")
        if delta.get("translated") is True:
            container.write("**Translation:** complete")
        if "word_counts" in delta:
            wc = delta.get("word_counts") or 0
            ner = delta.get("named_entities") or 0
            container.write(f"**Word analysis:** {wc} unique words, {ner} named entities")
        if delta.get("summary") is True:
            container.write("**Summary:** ready")
        flagged = delta.get("flagged")
        if isinstance(flagged, int) and flagged > 0:
            container.warning(f"**Hate speech:** {flagged} segment(s) flagged")


def _submit_files(
    client: BackendClient,
    uploaded_files: Sequence[Any],
    opts: dict[str, Any],
) -> list[dict[str, str]]:
    """Submit every uploaded file up front and return their job handles.

    Submitting the whole batch before tracking means the backend has
    queued every job, so a browser reload mid-run can re-discover and
    resume them via :meth:`BackendClient.list_jobs`.

    Args:
        client: Backend client bound to the caller's identity.
        uploaded_files: Streamlit ``UploadedFile`` objects.
        opts: Pipeline options matching :class:`JobOptions`.

    Returns:
        list[dict[str, str]]: ``[{"job_id": ..., "file_name": ...}, ...]``
            for every file the backend accepted, in upload order.
    """
    jobs: list[dict[str, str]] = []
    for index, uploaded_file in enumerate(uploaded_files, start=1):
        raw_name = str(getattr(uploaded_file, "name", f"File {index}"))
        file_name = Path(raw_name).name or f"File {index}"
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        # Hand the file object straight to the client so httpx streams it in
        # chunks; reading it into bytes here would hold a second full copy of
        # every file in memory on top of Streamlit's own upload buffer.
        try:
            job_id = client.submit_job(file_name, uploaded_file, opts)
        except Exception as exc:
            logger.exception("Failed to submit job for {}.", file_name)
            st.warning(f"Could not submit {file_name}: {exc}")
            continue
        jobs.append({"job_id": job_id, "file_name": file_name})
    return jobs


def _track_jobs(
    client: BackendClient,
    jobs: Sequence[dict[str, str]],
    results_container: Any | None = None,
) -> list[dict[str, Any]]:
    """Follow queued/running jobs to completion and collect their results.

    Re-attachable by design: a job that is still running (e.g. after a
    browser reload) replays its event history on subscribe, so the live
    progress view resumes where it left off; a job that already finished
    is read straight from its snapshot without re-running anything.

    Args:
        client: Backend client bound to the caller's identity.
        jobs: ``[{"job_id": ..., "file_name": ...}, ...]`` to follow.
        results_container: Optional Streamlit container for inline progress.

    Returns:
        list[dict[str, Any]]: Normalized result payloads, in input order.
    """
    total_files = len(jobs)
    total_stages = len(PIPELINE_STAGE_LABELS)
    progress_bar = st.progress(0.0, text="Preparing files…")
    results: list[dict[str, Any]] = []

    for file_index, job in enumerate(jobs, start=1):
        job_id = job["job_id"]
        file_name = job.get("file_name") or job_id

        file_status = None
        if results_container is not None:
            file_status = results_container.status(
                f"Processing {file_name} ({file_index}/{total_files})",
                expanded=True,
                state="running",
            )

        try:
            snapshot = client.get_snapshot(job_id)
        except Exception as exc:
            logger.exception("Failed to fetch snapshot for {}.", job_id)
            if file_status is not None:
                file_status.update(label=f"{file_name} — unavailable: {exc}", state="error", expanded=False)
            continue

        status = str(snapshot.get("status", ""))
        if status in {"queued", "running"}:
            try:
                for event in client.subscribe_events(job_id):
                    stage_index = int(event.data.get("stage_index", 0))
                    stage_name = str(event.data.get("stage") or "Processing")
                    progress_bar.progress(
                        progress_value(
                            file_index=file_index,
                            total_files=total_files,
                            stage_index=min(stage_index, total_stages - 1),
                            total_stages=total_stages,
                        ),
                        text=(f"Processing file {file_index}/{total_files}: {file_name} ({stage_name})"),
                    )
                    if file_status is not None and event.name == "stage_completed":
                        _render_event_delta(file_status, event)
                    if event.name in {"job_completed", "job_failed"}:
                        break
            except Exception as exc:
                logger.exception("SSE subscription failed for {}.", file_name)
                if file_status is not None:
                    file_status.update(label=f"{file_name} — stream failed: {exc}", state="error", expanded=False)
                continue
            try:
                snapshot = client.get_snapshot(job_id)
            except Exception as exc:
                logger.exception("Failed to fetch snapshot for {}.", job_id)
                if file_status is not None:
                    file_status.update(label=f"{file_name} — snapshot failed: {exc}", state="error", expanded=False)
                continue
            status = str(snapshot.get("status", ""))

        if status == "failed":
            error = snapshot.get("error") or "Unknown failure"
            if file_status is not None:
                file_status.update(label=f"{file_name} — failed: {error}", state="error", expanded=False)
            else:
                st.warning(f"File {file_index}/{total_files} failed — {file_name}: {error}")
            continue

        if status != "completed":
            # A still-queued/running job with no terminal event, or one that
            # was interrupted by a backend restart — surface and move on.
            if file_status is not None:
                file_status.update(label=f"{file_name} — {status or 'unavailable'}", state="error", expanded=False)
            continue

        file_result = _normalize_snapshot_result(snapshot)
        file_result["file_name"] = file_name
        if file_result.get("skipped"):
            if file_status is not None:
                file_status.update(
                    label=(f"{file_name} — skipped ({file_result.get('skip_reason', 'No processable content.')})"),
                    state="complete",
                    expanded=False,
                )
            else:
                st.warning(
                    f"File {file_index}/{total_files} skipped — "
                    f"{file_name}: {file_result.get('skip_reason', 'No processable content.')}"
                )
        elif file_status is not None:
            file_status.update(label=f"{file_name} — complete", state="complete", expanded=False)
        results.append(file_result)
        st.session_state["results"] = list(results)
        progress_bar.progress(
            file_index / total_files,
            text=f"Completed file {file_index}/{total_files}: {file_name}",
        )

    progress_bar.progress(1.0, text=f"Processed {total_files} file(s).")
    return results


def _finalize_results(results: list[dict[str, Any]]) -> None:
    """Store collected results in session state and report batch status.

    Args:
        results: Normalized result payloads from :func:`_track_jobs`.
    """
    if not results:
        st.error("No files could be processed.")
        return
    st.session_state["results"] = results
    st.session_state["result"] = results[0]
    st.session_state["selected_result_file"] = results[0]["file_name"]
    st.session_state.setdefault("results_timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
    skipped_count = sum(1 for r in results if r.get("skipped"))
    if skipped_count == len(results):
        st.warning("All files were skipped — no speech detected.")
    elif skipped_count > 0:
        st.success(
            f"Done! {len(results) - skipped_count} of {len(results)} file(s) produced results. "
            f"Select another tab to view."
        )
    else:
        st.success("Done! Select another tab to view results.")


def _resume_active_jobs() -> None:
    """Re-attach to this owner's jobs after a browser reload.

    A refresh clears ``st.session_state`` but the owner identity survives
    in the URL (``?owner=<id>``), so the backend can still list the
    caller's in-memory jobs. We follow them to completion — resuming the
    live progress view for any still running and re-rendering those
    already finished — then store the results so the tabs populate.
    """
    client = _get_client(_get_owner_id())
    try:
        discovered = client.list_jobs()
    except Exception:
        logger.exception("Could not list jobs while resuming after reload.")
        return
    if not discovered:
        return
    # ``list_jobs`` is newest-first; present in submission order for a
    # stable, deterministic layout across reloads.
    jobs = [{"job_id": str(j["job_id"]), "file_name": str(j.get("file_name", j["job_id"]))} for j in discovered]
    jobs.reverse()
    with st.expander("Resuming your jobs…", expanded=True):
        results_container = st.container()
    results = _track_jobs(client, jobs, results_container=results_container)
    _finalize_results(results)


def _results_archive_bytes(
    results: Sequence[dict[str, Any]],
    archive_stem: str,
) -> bytes:
    """Bundle per-job backend archives into one ZIP for the user download.

    Each backend ``archive.zip`` already nests outputs under
    ``{stem}/{stem}_{output}.{ext}``. We unpack and re-emit those entries
    under ``{archive_stem}/...`` so the layout matches what the in-process
    UI used to produce.

    Args:
        results: Stored result entries.
        archive_stem: Top-level directory name inside the ZIP.

    Returns:
        bytes: ZIP archive bytes ready to feed into ``st.download_button``.
    """
    cache_key = (id(results), archive_stem)
    cached = st.session_state.get("_results_archive")
    if isinstance(cached, tuple) and cached[0] == cache_key:
        return cast(bytes, cached[1])

    client = _get_client(_get_owner_id())
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as outer:
        for result in results:
            job_id = result.get("job_id")
            if not job_id:
                continue
            try:
                payload, _ = client.download_artifact(job_id, "archive.zip")
            except Exception:
                logger.exception("Failed to fetch archive for job {}.", job_id)
                continue
            with zipfile.ZipFile(io.BytesIO(payload)) as inner:
                for member in inner.namelist():
                    outer.writestr(
                        f"{archive_stem}/{member}",
                        inner.read(member),
                    )
    payload_bytes = buffer.getvalue()
    st.session_state["_results_archive"] = (cache_key, payload_bytes)
    return payload_bytes


def _docint_jsonl_bytes(
    results: Sequence[dict[str, Any]],
    archive_stem: str,
) -> tuple[bytes, str, str]:
    """Return a docint JSONL or ZIP payload for the current results.

    Single-result runs return a plain JSONL byte stream; multi-result runs
    bundle each transcript's JSONL into a ZIP under
    ``{archive_stem}/{stem}.jsonl``.

    Args:
        results: Stored result entries.
        archive_stem: Download filename stem used for the ZIP case.

    Returns:
        tuple[bytes, str, str]: ``(data, file_name, mime)`` for
            :func:`st.download_button`.
    """
    cache_key = (id(results), archive_stem)
    cached = st.session_state.get("_docint_jsonl")
    if isinstance(cached, tuple) and cached[0] == cache_key:
        return cast(tuple[bytes, str, str], cached[1])

    client = _get_client(_get_owner_id())
    per_file: list[tuple[str, bytes]] = []
    for result in results:
        job_id = result.get("job_id")
        if not job_id:
            continue
        try:
            payload, _ = client.download_artifact(job_id, "docint.jsonl")
        except Exception:
            continue
        if not payload:
            continue
        stem = Path(str(result.get("file_name", "result"))).stem or "result"
        per_file.append((stem, payload))

    if not per_file:
        materialized = (b"", f"{archive_stem}.jsonl", "application/x-ndjson")
    elif len(per_file) == 1:
        stem, payload = per_file[0]
        materialized = (payload, f"{stem}.docint.jsonl", "application/x-ndjson")
    else:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for stem, payload in per_file:
                zf.writestr(f"{archive_stem}/{stem}.jsonl", payload)
        materialized = (
            buffer.getvalue(),
            f"{archive_stem}_docint.zip",
            "application/zip",
        )

    st.session_state["_docint_jsonl"] = (cache_key, materialized)
    return materialized


def _start_page() -> None:
    """Render the pipeline settings form in the Parameters tab."""
    st.markdown("## ⚙️ Pipeline settings")

    uploaded_files = st.file_uploader(
        "Audio / video file(s)",
        type=["mp3", "m4a", "mp4", "mkv", "ogg", "wav", "webm"],
        accept_multiple_files=True,
    )

    batch_error = check_batch_within_limit(uploaded_files, _max_batch_bytes()) if uploaded_files else None
    if batch_error:
        st.error(batch_error)

    try:
        languages = _fetch_languages()
    except Exception as exc:
        st.error(f"Could not reach the Nextext backend: {exc}")
        languages = {"whisper": [], "target": []}

    src_lang_maps, src_lang_names = _split_language_lists(languages.get("whisper", []))
    trg_lang_maps, trg_lang_names = _split_language_lists(languages.get("target", []))
    default_trg_lang_index, default_trg_lang_code = default_target_language(
        language_maps=trg_lang_maps,
        language_names=trg_lang_names,
    )

    task = st.radio("Task", ["transcribe", "translate"], horizontal=True)
    col1, col2 = st.columns(2)
    with col1:
        src_lang_name = st.selectbox(
            "Source language",
            ["Detect language", *src_lang_names],
            index=0,
        )
        words = st.checkbox("Word-level analysis")
        summarization = st.checkbox("Summarisation")
        hate_speech = st.checkbox("Hate speech detection")
    with col2:
        trg_lang_name = st.selectbox(
            "Target language (for translate task)",
            trg_lang_names,
            index=default_trg_lang_index if trg_lang_names else 0,
        )
        speakers = st.number_input("Max speakers", 1, 10, value=1, step=1)

    src_lang_code = _name_to_code(src_lang_maps).get(src_lang_name)
    trg_lang_code = _name_to_code(trg_lang_maps).get(
        trg_lang_name,
        default_trg_lang_code,
    )

    run = st.button("▶️ Run", disabled=not uploaded_files or batch_error is not None)

    st.session_state["opts"] = dict(
        src_lang=src_lang_code,
        trg_lang=trg_lang_code,
        task=task,
        speakers=int(speakers),
        words=words,
        summarization=summarization,
        hate_speech=hate_speech,
    )

    if run and uploaded_files and not batch_error:
        for key in ("_results_archive", "_docint_jsonl", "results", "result"):
            st.session_state.pop(key, None)
        st.session_state["results_timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        client = _get_client(_get_owner_id())
        jobs = _submit_files(client, uploaded_files, st.session_state["opts"])
        st.session_state["active_jobs"] = jobs
        with st.expander("Processing progress", expanded=True):
            results_container = st.container()
        results = _track_jobs(client, jobs, results_container=results_container)
        _finalize_results(results)


def _render_transcript_tab(result: dict[str, Any]) -> None:
    """Render the Transcript tab body for one result.

    Args:
        result: Stored result payload for one file.
    """
    st.subheader("📜 Transcript")
    if "file_name" in result:
        st.caption(f"Showing results for `{result['file_name']}`")
    if result.get("skipped"):
        st.warning(result.get("skip_reason", "This file was skipped during processing."))
    st.dataframe(result["transcript"], hide_index=True)


def _render_summary_tab(result: dict[str, Any]) -> None:
    """Render the Summary tab body for one result.

    Args:
        result: Stored result payload for one file.
    """
    if result.get("skipped"):
        st.warning(result.get("skip_reason", "This file was skipped during processing."))
    elif not result.get("summary"):
        st.warning("Summary not requested.")
    else:
        st.subheader("📝 Summary")
        st.write(result["summary"])
        st.download_button(
            label="Download",
            data=result["summary"],
            file_name=download_file_name(result, "summary.txt"),
            mime="text/plain",
        )


def _render_word_tab(result: dict[str, Any]) -> None:
    """Render the Word-level Analysis tab body for one result.

    Args:
        result: Stored result payload for one file.
    """
    if result.get("skipped"):
        st.warning(result.get("skip_reason", "This file was skipped during processing."))
        return
    wc_df = result.get("word_counts")
    if not isinstance(wc_df, pd.DataFrame) or wc_df.empty:
        st.warning("Word statistics not requested.")
        return

    st.subheader("🔠 Top Words")
    chart = (
        alt.Chart(wc_df.reset_index())
        .mark_bar()
        .encode(
            x=alt.X("word:N", sort="-y", title="word"),
            y=alt.Y("count:Q", title="count"),
            tooltip=["word", "count"],
        )
        .properties(height=400)
    )
    st.altair_chart(chart, width="stretch")

    st.subheader("🧩 Named Entities")
    ner_df = result.get("named_entities")
    if isinstance(ner_df, pd.DataFrame) and not ner_df.empty:
        top15 = ner_df.sort_values("Frequency", ascending=False).head(15).copy()
        top15["Label"] = top15["Entity"] + " (" + top15["Category"] + ")"
        st.altair_chart(
            alt.Chart(top15)
            .mark_bar()
            .encode(
                x=alt.X("Label:N", sort="-y", title="Entity"),
                y=alt.Y("Frequency:Q", title="Frequency"),
                tooltip=["Category", "Entity", "Frequency"],
            )
            .properties(height=350),
            width="stretch",
        )
    else:
        st.info("No named entities found.")

    st.subheader("☁️ Word Cloud")
    if result.get("wordcloud_url") and result.get("job_id"):
        try:
            client = _get_client(_get_owner_id())
            png_bytes, _ = client.download_artifact(result["job_id"], "wordcloud.png")
            st.image(io.BytesIO(png_bytes))
        except Exception:
            st.info("Word cloud unavailable.")
    else:
        st.info("Not enough content for a word cloud.")


def _render_hate_tab(result: dict[str, Any]) -> None:
    """Render the Hate Speech tab body for one result.

    Args:
        result: Stored result payload for one file.
    """
    if result.get("skipped"):
        st.warning(result.get("skip_reason", "This file was skipped during processing."))
        return
    findings = result.get("hate_speech_findings")
    if findings is None:
        st.warning("Hate speech detection not requested.")
        return
    if not findings:
        st.success("No hate speech detected.")
        return
    st.subheader("🚨 Hate Speech Findings")
    for item in findings:
        with st.expander(f"{item.get('start', '')} - {str(item.get('category', '')).title()}"):
            st.write(f"**Reason:** {item.get('reason', '')}")
            st.write(f"**Flagged text:** {item.get('text', '')}")


def main() -> None:
    """Run the Streamlit app with tab-based navigation."""
    st.set_page_config(page_title="Nextext", layout="wide")
    # Establish the per-browser owner identity before any backend call.
    # The helper mints a UUID4 on first visit and stamps it into the
    # URL via ``st.query_params`` so the identity survives reloads and
    # browser-history navigations. There is no client-side bootstrap;
    # every run returns a usable id immediately.
    _ensure_owner_id()
    st.title("Nextext")
    st.subheader("Transcribe, translate, and analyze audio/video files")

    tab_params, tab_transcript, tab_summary, tab_words, tab_hate = st.tabs(
        ["Parameters", "Transcript", "Summary", "Word-level Analysis", "Hate Speech"]
    )

    with tab_params:
        _start_page()

    # Reload recovery: a refresh clears session state, but the owner id
    # survives in the URL — re-discover this owner's jobs from the backend
    # and resume tracking so an in-flight run is never lost on reload.
    if not st.session_state.get("results"):
        _resume_active_jobs()

    results = st.session_state.get("results")
    if results is None and "result" in st.session_state:
        results = [st.session_state["result"]]

    if not results:
        msg = (
            "After you upload one or more files and press **Run** in the Parameters tab, the results will appear here."
        )
        for tab in (tab_transcript, tab_summary, tab_words, tab_hate):
            with tab:
                st.info(msg)
        st.markdown(
            """
            <hr style="margin-top:2rem;margin-bottom:1rem;">
            <p style="text-align:center;">
                🔗 <a href="https://github.com/nos-tromo/Nextext" target="_blank">GitHub</a>
            </p>
            """,
            unsafe_allow_html=True,
        )
        return

    archive_timestamp = st.session_state.get("results_timestamp") or datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_stem = f"{archive_timestamp}_nextext_output"
    st.download_button(
        label="⬇️ Download all outputs (ZIP)",
        data=_results_archive_bytes(results, archive_stem),
        file_name=f"{archive_stem}.zip",
        mime="application/zip",
        help=(
            "Bundles every produced output (transcripts, summaries, word "
            "tables, named entities, word clouds, hate-speech findings) for "
            "every processed file."
        ),
    )

    docint_archive_stem = f"{archive_timestamp}_nextext_docint"
    docint_data, docint_file_name, docint_mime = _docint_jsonl_bytes(
        results,
        docint_archive_stem,
    )
    st.download_button(
        label="⬇️ Download docint-compatible output(JSONL)",
        data=docint_data,
        file_name=docint_file_name,
        mime=docint_mime,
        disabled=not docint_data,
        key="docint_jsonl_download",
        help=(
            "Sentence-level transcripts formatted as JSON Lines for "
            "ingestion by the docint RAG pipeline. One file per upload "
            "when multiple files were processed."
        ),
    )

    selected_file_name = None
    if len(results) > 1:
        file_names = result_file_names(results)
        current_file_name = st.session_state.get(
            "selected_result_file",
            file_names[0],
        )
        if current_file_name not in file_names:
            current_file_name = file_names[0]
        selected_file_name = st.selectbox(
            "Processed file",
            file_names,
            index=file_names.index(current_file_name),
        )
        st.session_state["selected_result_file"] = selected_file_name

    result = select_result(results, selected_file_name)

    with tab_transcript:
        _render_transcript_tab(result)
    with tab_summary:
        _render_summary_tab(result)
    with tab_words:
        _render_word_tab(result)
    with tab_hate:
        _render_hate_tab(result)

    st.markdown(
        """
        <hr style="margin-top:2rem;margin-bottom:1rem;">
        <p style="text-align:center;">
            🔗 <a href="https://github.com/nos-tromo/nextext" target="_blank">Nextext</a>
        </p>
        """,
        unsafe_allow_html=True,
    )


def cli() -> None:
    """Run the Streamlit app as if invoked from the command line.

    Sets up the command line arguments as if the user typed
    ``streamlit run nextext/frontend/app.py``.
    """
    sys.argv = ["streamlit", "run", __file__, *sys.argv[1:]]
    sys.exit(st_cli.main())


if __name__ == "__main__":
    main()
