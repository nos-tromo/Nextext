"""Materialize per-job artifacts (CSV/XLSX/PNG/TXT/JSONL/ZIP) from job state.

Artifacts are rendered on demand from the in-memory ``state.result``; they are
never written to disk.
"""

from __future__ import annotations

import io
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
    detected_raw = state.result.get("resolved_src_lang")
    detected_language = normalize_language_code(str(detected_raw)) if detected_raw else None
    task = state.result.get("task") or state.options.task
    return build_docint_jsonl(
        source_file=state.file_name,
        source_file_hash=state.source_file_hash or None,
        language=language,
        detected_language=detected_language,
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


def build_batch_docint_jsonl(states: list[JobState]) -> bytes:
    """Concatenate the docint JSONL of every job into one combined payload.

    Each per-job payload is already newline-terminated, so plain
    concatenation yields a valid NDJSON stream. Jobs that produced no docint
    records (empty/skipped transcripts) contribute nothing.

    Args:
        states: Jobs to combine, in the order they should appear.

    Returns:
        bytes: UTF-8 JSONL bytes spanning all jobs. ``b""`` when no job
            produced any records.
    """
    parts = [build_docint_jsonl_for_job(state) for state in states]
    return b"".join(part for part in parts if part)


def _unique_archive_folder(stem: str, seen: dict[str, int]) -> str:
    """Allocate a collision-free top-level folder name for a job's outputs.

    Args:
        stem: The job's file stem (may repeat across uploads).
        seen: Running base-stem -> count map, mutated in place.

    Returns:
        str: ``stem`` for the first occurrence, then ``stem_2``, ``stem_3``, …
    """
    base = stem or "result"
    count = seen.get(base, 0) + 1
    seen[base] = count
    return base if count == 1 else f"{base}_{count}"


def build_batch_archive(states: list[JobState]) -> bytes:
    """Bundle every job's outputs into one ZIP, nested per job.

    Each job's per-job archive (``{stem}/{stem}_{output}.{ext}``) is copied
    under a collision-free top-level folder so that uploads sharing a name do
    not overwrite one another. Jobs that produced no files are skipped.

    Args:
        states: Jobs to bundle, in the order they should appear.

    Returns:
        bytes: ZIP bytes. ``b""`` when no job produced any output.
    """
    buffer = io.BytesIO()
    seen: dict[str, int] = {}
    wrote_any = False
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as outer:
        for state in states:
            inner_bytes = build_archive_for_job(state)
            with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
                members = [info for info in inner.infolist() if not info.is_dir()]
                if not members:
                    continue
                folder = _unique_archive_folder(Path(state.file_name).stem, seen)
                for info in members:
                    # Rewrite the leading folder so identically named jobs stay
                    # distinct; the rest of the inner path is preserved.
                    _, separator, remainder = info.filename.partition("/")
                    inner_name = remainder if separator else info.filename
                    outer.writestr(f"{folder}/{inner_name}", inner.read(info.filename))
                    wrote_any = True
    if not wrote_any:
        return b""
    return buffer.getvalue()


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


def render_batch_artifact(states: list[JobState], name: str) -> tuple[bytes, str] | None:
    """Render one cross-job batch artifact from the caller's completed jobs.

    Args:
        states: Completed jobs owned by the caller, in display order.
        name: Batch artifact name (``docint.jsonl`` or ``archive.zip``).

    Returns:
        tuple[bytes, str] | None: ``(payload, content_type)`` on success.
            ``None`` when the name is unsupported or no job produced output
            (caller should respond with 404).
    """
    if name == "docint.jsonl":
        payload = build_batch_docint_jsonl(states)
        if not payload:
            return None
        return payload, _APP_NDJSON
    if name == "archive.zip":
        payload = build_batch_archive(states)
        if not payload:
            return None
        return payload, _APP_ZIP
    return None


BATCH_ARTIFACTS: frozenset[str] = frozenset({"docint.jsonl", "archive.zip"})
