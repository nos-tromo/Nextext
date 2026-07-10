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
from nextext.pipeline import normalize_language_code, transcript_txt_exports


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

    docint receives the original (untranslated) transcript text regardless of
    task — see :func:`nextext.core.docint_transcript.transcript_segments_from_df`
    — so ``language`` is always the resolved source language here, not
    ``result["transcript_language"]`` (which is the target language for a
    ``translate`` task and only describes the human-facing/downstream-analysis
    text).

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
    resolved_src_lang = state.result.get("resolved_src_lang")
    language = normalize_language_code(str(resolved_src_lang)) if resolved_src_lang else None
    return build_docint_jsonl(
        source_file=state.file_name,
        source_file_hash=state.source_file_hash or None,
        language=language,
        segments=segments,
    )


def _render_archive_members(state: JobState) -> dict[str, bytes]:
    """Render every produced output for a job as ``name -> bytes`` members.

    This is the costly part of archive assembly — XLSX serialization,
    word-cloud rasterization, docint JSONL — so the mapping is cached on the
    job and reused by both the per-job and batch ZIP builders. Keys are archive
    file names without any leading job folder, e.g. ``{stem}_transcript.csv``.
    Keyframes are the one exception: they nest under a ``keyframes/``
    subfolder (e.g. ``keyframes/frame_000.jpg``) so they never collide with
    the flat ``{stem}_...``-style keys above. This intentionally differs from
    the standalone ``keyframes.zip`` artifact, which uses flat names.

    Args:
        state: The completed job.

    Returns:
        dict[str, bytes]: Decompressed member payloads keyed by file name, in a
            stable insertion order. Empty when the job produced no outputs.
    """
    if state.archive_members_cache is not None:
        return state.archive_members_cache

    stem = Path(state.file_name).stem or "result"
    result = state.result
    members: dict[str, bytes] = {}

    transcript = result.get("transcript")
    if isinstance(transcript, pd.DataFrame) and not transcript.empty:
        members[f"{stem}_transcript.csv"] = transcript.to_csv(index=False).encode("utf-8")
        members[f"{stem}_transcript.xlsx"] = _df_to_xlsx(transcript)
        for label, tsv in transcript_txt_exports(transcript):
            members[f"{stem}_{label}.txt"] = tsv.encode("utf-8")

    summary = result.get("summary")
    if isinstance(summary, str) and summary.strip():
        members[f"{stem}_summary.txt"] = summary.encode("utf-8")

    word_counts = result.get("word_counts")
    if isinstance(word_counts, pd.DataFrame) and not word_counts.empty:
        members[f"{stem}_words.csv"] = word_counts.to_csv(index=False).encode("utf-8")
        members[f"{stem}_words.xlsx"] = _df_to_xlsx(word_counts)

    named_entities = result.get("named_entities")
    if isinstance(named_entities, pd.DataFrame) and not named_entities.empty:
        members[f"{stem}_entities.csv"] = named_entities.to_csv(index=False).encode("utf-8")
        members[f"{stem}_entities.xlsx"] = _df_to_xlsx(named_entities)

    wordcloud = result.get("wordcloud")
    if isinstance(wordcloud, Figure):
        members[f"{stem}_wordcloud.png"] = _figure_to_png(wordcloud)

    findings = result.get("hate_speech_findings")
    if findings:
        findings_df = pd.DataFrame(findings)
        members[f"{stem}_hate_speech.csv"] = findings_df.to_csv(index=False).encode("utf-8")
        members[f"{stem}_hate_speech.xlsx"] = _df_to_xlsx(findings_df)

    docint = build_docint_jsonl_for_job(state)
    if docint:
        members[f"{stem}_docint.jsonl"] = docint

    frames = result.get("keyframes")
    if frames:
        for index, payload in enumerate(frames):
            members[f"keyframes/frame_{index:03d}.jpg"] = payload

    state.archive_members_cache = members
    return members


def _zip_members(named_members: list[tuple[str, bytes]]) -> bytes:
    """Compress ``(archive_path, payload)`` pairs into one ZIP.

    Args:
        named_members: Ordered ``(path, bytes)`` pairs. Paths must be unique
            within the archive; their order is preserved in the output.

    Returns:
        bytes: ``ZIP_DEFLATED`` archive bytes.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path, payload in named_members:
            zf.writestr(path, payload)
    return buffer.getvalue()


def build_archive_for_job(state: JobState) -> bytes:
    """Bundle every produced output for the job into a ZIP archive.

    The expensive render is cached on the job (see
    :func:`_render_archive_members`); only the cheap ZIP compression is repeated
    if the per-job archive is downloaded more than once.

    Args:
        state: The completed job.

    Returns:
        bytes: ZIP bytes laid out as ``{stem}/{stem}_{output}.{ext}``.
    """
    stem = Path(state.file_name).stem or "result"
    members = _render_archive_members(state)
    return _zip_members([(f"{stem}/{name}", payload) for name, payload in members.items()])


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


_BATCH_SUMMARIES_NAME = "batch_summaries.txt"
_SUMMARY_BANNER_RULE = "=" * 40


def _build_batch_summaries_txt(states: list[JobState]) -> bytes:
    """Aggregate every job's summary into one banner-delimited text file.

    Each job whose ``result["summary"]`` is a non-empty string contributes one
    block: a banner header carrying the upload's file name (with extension), a
    blank line, then the summary text. Jobs without a summary are skipped, so
    the file lists exactly the summarized files in ``states`` order. This is a
    convenience index for the batch ZIP; the per-job ``{stem}_summary.txt``
    members are still emitted separately by :func:`_render_archive_members`.

    Args:
        states: Jobs to combine, in the order they should appear.

    Returns:
        bytes: UTF-8 text bytes, or ``b""`` when no job produced a summary.
    """
    blocks: list[str] = []
    for state in states:
        summary = state.result.get("summary")
        if not (isinstance(summary, str) and summary.strip()):
            continue
        header = f"{_SUMMARY_BANNER_RULE}\n{state.file_name}\n{_SUMMARY_BANNER_RULE}"
        blocks.append(f"{header}\n\n{summary.strip()}\n")
    if not blocks:
        return b""
    return "\n".join(blocks).encode("utf-8")


def build_batch_archive(states: list[JobState]) -> bytes:
    """Bundle every job's outputs into one ZIP, nested per job.

    Each job's rendered members (``{stem}_{output}.{ext}``) are placed under a
    collision-free top-level folder so that uploads sharing a name do not
    overwrite one another. Members come straight from the per-job render cache
    and are compressed exactly once here — unlike the old nested layout, no
    per-job archive is decompressed and re-compressed. Jobs that produced no
    files are skipped.

    When at least one job carries a summary, a single top-level
    ``batch_summaries.txt`` manifest (see :func:`_build_batch_summaries_txt`) is
    added at the archive root, listing every summarized file's name and summary
    text; it is omitted entirely when no job has a summary.

    Args:
        states: Jobs to bundle, in the order they should appear.

    Returns:
        bytes: ZIP bytes. ``b""`` when no job produced any output.
    """
    seen: dict[str, int] = {}
    named: list[tuple[str, bytes]] = []
    summaries = _build_batch_summaries_txt(states)
    if summaries:
        named.append((_BATCH_SUMMARIES_NAME, summaries))
    for state in states:
        members = _render_archive_members(state)
        if not members:
            continue
        folder = _unique_archive_folder(Path(state.file_name).stem, seen)
        named.extend((f"{folder}/{name}", payload) for name, payload in members.items())
    if not named:
        return b""
    return _zip_members(named)


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
    if name == "transcript.txt":
        if _missing_dataframe(state, "transcript"):
            return None
        exports = dict(transcript_txt_exports(result["transcript"]))
        return exports["transcript"].encode("utf-8"), _TEXT_PLAIN
    if name == "translation.txt":
        if _missing_dataframe(state, "transcript"):
            return None
        exports = dict(transcript_txt_exports(result["transcript"]))
        translation_tsv = exports.get("translation")
        if translation_tsv is None:
            return None
        return translation_tsv.encode("utf-8"), _TEXT_PLAIN
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
    if name == "keyframes.zip":
        frames = result.get("keyframes")
        if not frames:
            return None
        members = [(f"frame_{index:03d}.jpg", payload) for index, payload in enumerate(frames)]
        return _zip_members(members), _APP_ZIP
    return None


SUPPORTED_ARTIFACTS: frozenset[str] = frozenset(
    {
        "transcript.csv",
        "transcript.xlsx",
        "transcript.txt",
        "translation.txt",
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
        "keyframes.zip",
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
