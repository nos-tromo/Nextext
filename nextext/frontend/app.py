"""Streamlit UI for the Nextext audio analysis workflow.

The UI is a thin HTTP client over the FastAPI backend defined in
``nextext.api``. All heavy ML work happens server-side; this module
intentionally avoids importing ``nextext.pipeline`` or any module that
chains to it, so the Streamlit Docker image can ship with only the
``frontend`` dependency group installed.
"""

from __future__ import annotations

import io
import sys
import zipfile
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import altair as alt
import pandas as pd  # type: ignore[import-untyped]
import streamlit as st
from loguru import logger
from streamlit.web import cli as st_cli

from nextext.frontend.client import BackendClient, StageEvent
from nextext.frontend.state import (
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


@st.cache_resource(show_spinner=False)
def _get_client() -> BackendClient:
    """Return a session-cached :class:`BackendClient` instance.

    Returns:
        BackendClient: A long-lived client reused across reruns.
    """
    return BackendClient()


@st.cache_data(show_spinner=False, ttl=300)
def _fetch_languages() -> dict[str, list[dict[str, str]]]:
    """Fetch the language mappings once per session.

    Returns:
        dict[str, list[dict[str, str]]]: ``{"whisper": [...], "target": [...]}``.
    """
    client = _get_client()
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
        "named_entities": named_entities_list_to_dataframe(
            api_result.get("named_entities")
        ),
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
            container.write(
                f"**Word analysis:** {wc} unique words, {ner} named entities"
            )
        if delta.get("summary") is True:
            container.write("**Summary:** ready")
        flagged = delta.get("flagged")
        if isinstance(flagged, int) and flagged > 0:
            container.warning(f"**Hate speech:** {flagged} segment(s) flagged")


def _process_uploaded_files(
    uploaded_files: Sequence[Any],
    opts: dict[str, Any],
    results_container: Any | None = None,
) -> list[dict[str, Any]]:
    """Submit each uploaded file as a backend job and collect snapshots.

    Args:
        uploaded_files: Streamlit ``UploadedFile`` objects.
        opts: Pipeline options matching :class:`JobOptions`.
        results_container: Optional Streamlit container for inline progress.

    Returns:
        list[dict[str, Any]]: Result payloads in upload order.
    """
    client = _get_client()
    total_files = len(uploaded_files)
    total_stages = len(PIPELINE_STAGE_LABELS)
    progress_bar = st.progress(0.0, text="Preparing files…")
    results: list[dict[str, Any]] = []

    for file_index, uploaded_file in enumerate(uploaded_files, start=1):
        raw_name = str(getattr(uploaded_file, "name", f"File {file_index}"))
        file_name = Path(raw_name).name or f"File {file_index}"

        file_status = None
        if results_container is not None:
            file_status = results_container.status(
                f"Processing {file_name} ({file_index}/{total_files})",
                expanded=True,
                state="running",
            )

        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        payload = uploaded_file.read()
        try:
            job_id = client.submit_job(file_name, payload, opts)
        except Exception as exc:  # noqa: BLE001 - surface any submit failure
            logger.exception("Failed to submit job for {}.", file_name)
            if file_status is not None:
                file_status.update(
                    label=f"{file_name} — submit failed: {exc}",
                    state="error",
                    expanded=False,
                )
            continue

        terminal: StageEvent | None = None
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
                    text=(
                        f"Processing file {file_index}/{total_files}: "
                        f"{file_name} ({stage_name})"
                    ),
                )
                if file_status is not None and event.name == "stage_completed":
                    _render_event_delta(file_status, event)
                if event.name in {"job_completed", "job_failed"}:
                    terminal = event
                    break
        except Exception as exc:  # noqa: BLE001
            logger.exception("SSE subscription failed for {}.", file_name)
            if file_status is not None:
                file_status.update(
                    label=f"{file_name} — stream failed: {exc}",
                    state="error",
                    expanded=False,
                )
            continue

        if terminal is None or terminal.name == "job_failed":
            error = (
                terminal.data.get("error")
                if terminal is not None
                else "Unknown failure"
            )
            if file_status is not None:
                file_status.update(
                    label=f"{file_name} — failed: {error}",
                    state="error",
                    expanded=False,
                )
            else:
                st.warning(
                    f"File {file_index}/{total_files} failed — {file_name}: {error}"
                )
            continue

        try:
            snapshot = client.get_snapshot(job_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to fetch snapshot for {}.", job_id)
            if file_status is not None:
                file_status.update(
                    label=f"{file_name} — snapshot failed: {exc}",
                    state="error",
                    expanded=False,
                )
            continue

        file_result = _normalize_snapshot_result(snapshot)
        file_result["file_name"] = file_name
        if file_result.get("skipped"):
            if file_status is not None:
                file_status.update(
                    label=(
                        f"{file_name} — skipped "
                        f"({file_result.get('skip_reason', 'No processable content.')})"
                    ),
                    state="complete",
                    expanded=False,
                )
            else:
                st.warning(
                    f"File {file_index}/{total_files} skipped — "
                    f"{file_name}: {file_result.get('skip_reason', 'No processable content.')}"
                )
        elif file_status is not None:
            file_status.update(
                label=f"{file_name} — complete",
                state="complete",
                expanded=False,
            )
        results.append(file_result)
        st.session_state["results"] = list(results)
        progress_bar.progress(
            file_index / total_files,
            text=f"Completed file {file_index}/{total_files}: {file_name}",
        )

    progress_bar.progress(1.0, text=f"Processed {total_files} file(s).")
    return results


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

    client = _get_client()
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

    client = _get_client()
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

    try:
        languages = _fetch_languages()
    except Exception as exc:  # noqa: BLE001
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
            ["Detect language"] + src_lang_names,
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

    persist = st.checkbox(
        "Save results across browser sessions",
        value=False,
        help=(
            "Stored on the server until you delete them via the Saved jobs "
            "sidebar. Leave this off when the audio is sensitive — by "
            "default, results live only in memory and disappear after the "
            "configured TTL."
        ),
    )

    src_lang_code = _name_to_code(src_lang_maps).get(src_lang_name)
    trg_lang_code = _name_to_code(trg_lang_maps).get(
        trg_lang_name,
        default_trg_lang_code,
    )

    run = st.button("▶️ Run", disabled=not uploaded_files)

    st.session_state["opts"] = dict(
        src_lang=src_lang_code,
        trg_lang=trg_lang_code,
        task=task,
        speakers=int(speakers),
        words=words,
        summarization=summarization,
        hate_speech=hate_speech,
        persist=persist,
    )

    if run and uploaded_files:
        st.session_state.pop("_results_archive", None)
        st.session_state.pop("_docint_jsonl", None)
        with st.expander("Processing progress", expanded=True):
            results_container = st.container()
        results = _process_uploaded_files(
            uploaded_files,
            st.session_state["opts"],
            results_container=results_container,
        )
        if results:
            st.session_state["results"] = results
            st.session_state["result"] = results[0]
            st.session_state["selected_result_file"] = results[0]["file_name"]
            st.session_state["results_timestamp"] = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
            skipped_count = sum(1 for r in results if r.get("skipped"))
            if skipped_count == len(results):
                st.warning("All files were skipped — no speech detected.")
            elif skipped_count > 0:
                st.success(
                    f"Done! {len(results) - skipped_count} of {len(results)} "
                    f"file(s) produced results. Select another tab to view."
                )
            else:
                st.success("Done! Select another tab to view results.")
        else:
            st.error("No files could be processed.")


def _render_transcript_tab(result: dict[str, Any]) -> None:
    """Render the Transcript tab body for one result.

    Args:
        result: Stored result payload for one file.
    """
    st.subheader("📜 Transcript")
    if "file_name" in result:
        st.caption(f"Showing results for `{result['file_name']}`")
    if result.get("skipped"):
        st.warning(
            result.get("skip_reason", "This file was skipped during processing.")
        )
    st.dataframe(result["transcript"], hide_index=True)


def _render_summary_tab(result: dict[str, Any]) -> None:
    """Render the Summary tab body for one result.

    Args:
        result: Stored result payload for one file.
    """
    if result.get("skipped"):
        st.warning(
            result.get("skip_reason", "This file was skipped during processing.")
        )
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
        st.warning(
            result.get("skip_reason", "This file was skipped during processing.")
        )
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
            client = _get_client()
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
        st.warning(
            result.get("skip_reason", "This file was skipped during processing.")
        )
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
        with st.expander(
            f"{item.get('start', '')} – {str(item.get('category', '')).title()}"
        ):
            st.write(f"**Reason:** {item.get('reason', '')}")
            st.write(f"**Flagged text:** {item.get('text', '')}")


def _load_saved_job(job_id: str) -> None:
    """Load a persisted snapshot into ``st.session_state["results"]``.

    Used by the Saved-jobs sidebar to re-open a previously stored run
    inside the result tabs without re-running the pipeline.

    Args:
        job_id: Job identifier returned by ``client.list_jobs``.
    """
    try:
        snapshot = _get_client().get_snapshot(job_id)
    except Exception as exc:  # noqa: BLE001
        st.sidebar.error(f"Could not load job {job_id}: {exc}")
        return
    if snapshot.get("status") != "completed":
        st.sidebar.warning(
            f"Job {snapshot.get('file_name', job_id)} is "
            f"{snapshot.get('status', 'unknown')}; only completed jobs can be reloaded."
        )
        return
    result = _normalize_snapshot_result(snapshot)
    result["file_name"] = snapshot.get("file_name") or job_id
    st.session_state["results"] = [result]
    st.session_state["result"] = result
    st.session_state["selected_result_file"] = result["file_name"]
    st.session_state["results_timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state.pop("_results_archive", None)
    st.session_state.pop("_docint_jsonl", None)


def _render_saved_jobs_sidebar() -> None:
    """Populate the sidebar with the caller's persistent jobs.

    Lets users reopen previously-saved work and delete entries they no
    longer need. Silently skips when the backend is unreachable so the
    main UI stays usable.
    """
    with st.sidebar:
        st.markdown("### 💾 Saved jobs")
        try:
            jobs = _get_client().list_jobs()
        except Exception:  # noqa: BLE001
            st.caption("Backend unreachable — saved jobs unavailable.")
            return
        if not jobs:
            st.caption(
                "Tick **Save results across browser sessions** before "
                "running to keep results here."
            )
            return
        for job in jobs:
            status_label = str(job.get("status", "unknown"))
            file_name = str(job.get("file_name", job["job_id"]))
            label = f"📄 {file_name} — {status_label}"
            cols = st.columns([4, 1])
            with cols[0]:
                if st.button(label, key=f"saved_open_{job['job_id']}"):
                    _load_saved_job(job["job_id"])
                    st.rerun()
            with cols[1]:
                if st.button("🗑", key=f"saved_delete_{job['job_id']}"):
                    try:
                        _get_client().delete_job(job["job_id"])
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Delete failed: {exc}")
                    else:
                        st.rerun()


def main() -> None:
    """Run the Streamlit app with tab-based navigation."""
    st.set_page_config(page_title="Nextext", layout="wide")
    st.title("Nextext")
    st.subheader("Transcribe, translate, and analyze audio/video files")

    _render_saved_jobs_sidebar()

    tab_params, tab_transcript, tab_summary, tab_words, tab_hate = st.tabs(
        ["Parameters", "Transcript", "Summary", "Word-level Analysis", "Hate Speech"]
    )

    with tab_params:
        _start_page()

    results = st.session_state.get("results")
    if results is None and "result" in st.session_state:
        results = [st.session_state["result"]]

    if not results:
        msg = (
            "After you upload one or more files and press **Run** in the "
            "Parameters tab, the results will appear here."
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

    archive_timestamp = st.session_state.get(
        "results_timestamp"
    ) or datetime.now().strftime("%Y%m%d_%H%M%S")
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
            🔗 <a href="https://github.com/nos-tromo/nextext" target="_blank">GitHub&nbsp;Repository</a>
        </p>
        """,
        unsafe_allow_html=True,
    )


def cli() -> None:
    """Run the Streamlit app as if invoked from the command line.

    Sets up the command line arguments as if the user typed
    ``streamlit run nextext/frontend/app.py``.
    """
    sys.argv = ["streamlit", "run", __file__] + sys.argv[1:]
    sys.exit(st_cli.main())


if __name__ == "__main__":
    main()
