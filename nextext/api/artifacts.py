"""Materialize per-job artifacts (CSV/XLSX/PNG/TXT/JSONL/ZIP) from job state.

These helpers port the ZIP-archive builders that previously lived in
``nextext/app.py`` (lines 66-281) so the API can serve byte payloads
directly without rebuilding them in the Streamlit frontend.

For persistent jobs that were rehydrated from disk at backend startup,
``state.result`` is empty until the first artifact download triggers a
lazy load via :func:`hydrate_from_disk`. The same renderers then run
against the freshly-populated in-memory dict, so the rest of the module
does not need to special-case persisted vs. live jobs.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any, cast

import pandas as pd
from matplotlib.figure import Figure

from nextext.api.jobs import JobState
from nextext.core.docint_transcript import (
    build_docint_jsonl,
    transcript_segments_from_df,
)
from nextext.pipeline import normalize_language_code


def hydrate_from_disk(state: JobState) -> None:
    """Populate ``state.result`` from a persistent job's artifact store.

    No-op when the job is ephemeral, when the store is unavailable, or
    when the result has already been materialised. Safe to call multiple
    times.

    Args:
        state: The job whose result should be available in memory.
    """
    if not state.persistent or state.artifact_store is None:
        return
    if state.result:
        return
    store = state.artifact_store
    if not store.root.is_dir():
        return

    meta_path = store.path("meta.json")
    meta: dict[str, Any] = {}
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}

    result: dict[str, Any] = {
        "transcript_language": meta.get("transcript_language"),
        "resolved_src_lang": meta.get("resolved_src_lang"),
        "task": meta.get("task", state.options.task),
        "summary": meta.get("summary"),
        "hate_speech_findings": meta.get("hate_speech_findings"),
        "skipped": bool(meta.get("skipped", False)),
        "skip_reason": meta.get("skip_reason"),
    }

    transcript_path = store.path("transcript.parquet")
    if transcript_path.is_file():
        result["transcript"] = pd.read_parquet(transcript_path)
    else:
        result["transcript"] = pd.DataFrame()

    word_counts_path = store.path("word_counts.parquet")
    if word_counts_path.is_file():
        result["word_counts"] = pd.read_parquet(word_counts_path)

    named_entities_path = store.path("named_entities.parquet")
    if named_entities_path.is_file():
        result["named_entities"] = pd.read_parquet(named_entities_path)

    findings_path = store.path("hate_speech.parquet")
    if findings_path.is_file():
        result["hate_speech_findings"] = pd.read_parquet(findings_path).to_dict(orient="records")

    summary_path = store.path("summary.txt")
    if summary_path.is_file():
        result["summary"] = summary_path.read_text(encoding="utf-8")

    if store.exists("wordcloud.png"):
        result["_wordcloud_url"] = f"/api/v1/jobs/{state.job_id}/artifacts/wordcloud.png"

    state.result = result


def _df_to_xlsx(df: pd.DataFrame) -> bytes:
    """Serialize a DataFrame to XLSX bytes using ``openpyxl``.

    Args:
        df: DataFrame to serialize.

    Returns:
        bytes: XLSX bytes ready for streaming to clients.
    """
    buffer = io.BytesIO()
    with pd.ExcelWriter(cast(Any, buffer), engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1")
    return buffer.getvalue()


def _figure_to_png(figure: Figure) -> bytes:
    """Render a Matplotlib ``Figure`` to PNG bytes.

    Args:
        figure: A Matplotlib figure.

    Returns:
        bytes: PNG bytes.
    """
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight")
    return buffer.getvalue()


def build_docint_jsonl_for_job(state: JobState) -> bytes:
    """Render the docint-flavoured JSONL payload for one job.

    Args:
        state: The job whose transcript should be serialized.

    Returns:
        bytes: UTF-8 JSONL bytes. Empty when the transcript has no segments.
    """
    transcript = state.result.get("transcript")
    if not isinstance(transcript, pd.DataFrame) or transcript.empty:
        return b""
    segments = transcript_segments_from_df(transcript)
    if not segments:
        return b""
    transcript_language = state.result.get("transcript_language") or state.result.get("resolved_src_lang")
    language = normalize_language_code(str(transcript_language)) if transcript_language else None
    task = state.result.get("task") or state.options.task
    return build_docint_jsonl(
        source_file=state.file_name,
        source_file_hash=state.source_file_hash or None,
        language=language,
        task=task,
        segments=segments,
    )


def build_archive_for_job(state: JobState) -> bytes:
    """Bundle every produced output for the job into a ZIP archive.

    Args:
        state: The completed job.

    Returns:
        bytes: ZIP bytes laid out as ``{stem}/{stem}_{output}.{ext}``.
    """
    if state.archive_cache is not None:
        return state.archive_cache

    stem = Path(state.file_name).stem or "result"
    base = f"{stem}/{stem}"
    result = state.result

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        transcript = result.get("transcript")
        if isinstance(transcript, pd.DataFrame) and not transcript.empty:
            zf.writestr(
                f"{base}_transcript.csv",
                transcript.to_csv(index=False).encode("utf-8"),
            )
            zf.writestr(f"{base}_transcript.xlsx", _df_to_xlsx(transcript))

        summary = result.get("summary")
        if isinstance(summary, str) and summary.strip():
            zf.writestr(f"{base}_summary.txt", summary.encode("utf-8"))

        word_counts = result.get("word_counts")
        if isinstance(word_counts, pd.DataFrame) and not word_counts.empty:
            zf.writestr(
                f"{base}_words.csv",
                word_counts.to_csv(index=False).encode("utf-8"),
            )
            zf.writestr(f"{base}_words.xlsx", _df_to_xlsx(word_counts))

        named_entities = result.get("named_entities")
        if isinstance(named_entities, pd.DataFrame) and not named_entities.empty:
            zf.writestr(
                f"{base}_entities.csv",
                named_entities.to_csv(index=False).encode("utf-8"),
            )
            zf.writestr(f"{base}_entities.xlsx", _df_to_xlsx(named_entities))

        wordcloud = result.get("wordcloud")
        if isinstance(wordcloud, Figure):
            zf.writestr(f"{base}_wordcloud.png", _figure_to_png(wordcloud))
        elif state.persistent and state.artifact_store is not None and state.artifact_store.exists("wordcloud.png"):
            # Rehydrated persistent jobs no longer hold the Figure in
            # memory; serve the PNG bytes that were written to disk at
            # completion time so archive.zip stays complete.
            zf.writestr(
                f"{base}_wordcloud.png",
                state.artifact_store.path("wordcloud.png").read_bytes(),
            )

        findings = result.get("hate_speech_findings")
        if findings:
            findings_df = pd.DataFrame(findings)
            zf.writestr(
                f"{base}_hate_speech.csv",
                findings_df.to_csv(index=False).encode("utf-8"),
            )
            zf.writestr(f"{base}_hate_speech.xlsx", _df_to_xlsx(findings_df))

        docint = build_docint_jsonl_for_job(state)
        if docint:
            zf.writestr(f"{base}_docint.jsonl", docint)

    state.archive_cache = buffer.getvalue()
    return state.archive_cache


def _missing_dataframe(state: JobState, key: str) -> bool:
    """Return ``True`` when the result dict has no usable DataFrame for ``key``.

    Args:
        state: Source job state.
        key: Result dict key, e.g. ``transcript``.

    Returns:
        bool: Whether the artifact would be empty.
    """
    df = state.result.get(key)
    return not (isinstance(df, pd.DataFrame) and not df.empty)


_TEXT_CSV = "text/csv; charset=utf-8"
_TEXT_PLAIN = "text/plain; charset=utf-8"
_APP_XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
_APP_PNG = "image/png"
_APP_NDJSON = "application/x-ndjson"
_APP_ZIP = "application/zip"


def render_artifact(state: JobState, name: str) -> tuple[bytes, str] | None:
    """Render one artifact for a completed job.

    Args:
        state: The job whose results back the artifact.
        name: Artifact name, e.g. ``transcript.csv`` or ``archive.zip``.

    Returns:
        tuple[bytes, str] | None: ``(payload, content_type)`` on success.
            ``None`` when the artifact is undefined or the underlying result
            was not produced (caller should respond with 404).
    """
    hydrate_from_disk(state)
    if name == "wordcloud.png" and state.persistent and state.artifact_store is not None:
        # The wordcloud is the only artifact that lives on disk as PNG
        # rather than being rebuilt from a matplotlib ``Figure``; serve
        # the bytes directly when they're available.
        if state.artifact_store.exists("wordcloud.png"):
            payload = state.artifact_store.path("wordcloud.png").read_bytes()
            return payload, _APP_PNG
    result = state.result
    if name == "transcript.csv":
        if _missing_dataframe(state, "transcript"):
            return None
        return (
            cast(pd.DataFrame, result["transcript"]).to_csv(index=False).encode("utf-8"),
            _TEXT_CSV,
        )
    if name == "transcript.xlsx":
        if _missing_dataframe(state, "transcript"):
            return None
        return _df_to_xlsx(result["transcript"]), _APP_XLSX
    if name == "summary.txt":
        summary = result.get("summary")
        if not (isinstance(summary, str) and summary.strip()):
            return None
        return summary.encode("utf-8"), _TEXT_PLAIN
    if name == "wordcounts.csv":
        if _missing_dataframe(state, "word_counts"):
            return None
        return (
            cast(pd.DataFrame, result["word_counts"]).to_csv(index=False).encode("utf-8"),
            _TEXT_CSV,
        )
    if name == "wordcounts.xlsx":
        if _missing_dataframe(state, "word_counts"):
            return None
        return _df_to_xlsx(result["word_counts"]), _APP_XLSX
    if name == "entities.csv":
        if _missing_dataframe(state, "named_entities"):
            return None
        return (
            cast(pd.DataFrame, result["named_entities"]).to_csv(index=False).encode("utf-8"),
            _TEXT_CSV,
        )
    if name == "entities.xlsx":
        if _missing_dataframe(state, "named_entities"):
            return None
        return _df_to_xlsx(result["named_entities"]), _APP_XLSX
    if name == "wordcloud.png":
        wordcloud = result.get("wordcloud")
        if not isinstance(wordcloud, Figure):
            return None
        return _figure_to_png(wordcloud), _APP_PNG
    if name == "hate_speech.csv":
        findings = result.get("hate_speech_findings")
        if not findings:
            return None
        df = pd.DataFrame(findings)
        return df.to_csv(index=False).encode("utf-8"), _TEXT_CSV
    if name == "hate_speech.xlsx":
        findings = result.get("hate_speech_findings")
        if not findings:
            return None
        return _df_to_xlsx(pd.DataFrame(findings)), _APP_XLSX
    if name == "docint.jsonl":
        payload = build_docint_jsonl_for_job(state)
        if not payload:
            return None
        return payload, _APP_NDJSON
    if name == "archive.zip":
        return build_archive_for_job(state), _APP_ZIP
    return None


SUPPORTED_ARTIFACTS: frozenset[str] = frozenset(
    {
        "transcript.csv",
        "transcript.xlsx",
        "summary.txt",
        "wordcounts.csv",
        "wordcounts.xlsx",
        "entities.csv",
        "entities.xlsx",
        "wordcloud.png",
        "hate_speech.csv",
        "hate_speech.xlsx",
        "docint.jsonl",
        "archive.zip",
    }
)
