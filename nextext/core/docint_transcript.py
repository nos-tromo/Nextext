"""JSONL export of structured transcripts for the docint ingestion pipeline.

docint (the sibling RAG app) no longer runs its own Whisper-backed audio
reader; it ingests structured transcripts produced by Nextext. This module
owns the serialization format so the wire contract lives next to the
Whisper/pyannote pipeline that produces it.

The payload is UTF-8 JSON Lines — one JSON object per sentence-level
segment, newline terminated (including the final line). Optional keys are
omitted entirely when unset so the docint side can rely on key presence
rather than ``None`` sentinels.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
import pycountry

from nextext.pipeline import normalize_language_code

__all__ = [
    "build_docint_jsonl",
    "format_hhmmss",
    "language_name",
    "parse_hhmmss_to_seconds",
    "transcript_segments_from_df",
]


def language_name(lang_code: str | None) -> str:
    """Convert an ISO language code to a human-readable name for LLM output settings.

    Args:
        lang_code (str | None): The ISO 639-1 language code.

    Returns:
        str: The human-readable language name, or ``"German"`` if the code
            is ``None``.
    """
    if not lang_code:
        return "German"
    lang = pycountry.languages.get(alpha_2=normalize_language_code(lang_code))
    return lang.name if lang is not None else lang_code


def format_hhmmss(seconds: float) -> str:
    """Format a seconds offset as a zero-padded ``HH:MM:SS`` string.

    In the current pipeline the caller always passes integer-valued floats
    (seconds derived from :func:`parse_hhmmss_to_seconds`, which itself
    reads the already-rounded ``HH:MM:SS`` strings produced by
    :func:`nextext.core.transcription._seconds_to_time`). The
    ``int(float(seconds))`` conversion below truncates fractional input,
    but because the input is integer-valued floats the truncate-vs-round
    distinction is a no-op for the current call sites. The
    :func:`int(float(...))` floor is kept only as a safety net for callers
    that might feed genuine fractions in the future.

    Args:
        seconds (float): A non-negative seconds offset into the audio.

    Returns:
        str: The offset rendered as ``HH:MM:SS`` with at least two digits
            for each component (hours may grow beyond two digits for very
            long recordings).

    Raises:
        ValueError: If ``seconds`` is negative.
    """
    if seconds < 0:
        raise ValueError("seconds must be non-negative.")
    total = int(float(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    # ``str(timedelta(...))`` would emit ``0:00:05`` (single-digit hour) for
    # anything under 10 hours; the docint ingest expects a zero-padded form
    # for stable lexicographic sorting, so we format manually.
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def parse_hhmmss_to_seconds(value: Any) -> float:
    """Parse a ``HH:MM:SS`` / ``H:MM:SS`` / ``D day, H:MM:SS`` string to seconds.

    The transcript DataFrame stores timestamps as the output of
    :func:`datetime.timedelta.__str__`, which drops the hour leading zero
    and optionally prefixes a ``N day,`` component for recordings longer
    than 24 hours. Numeric inputs (ints/floats) pass through unchanged so
    upstream callers can feed either representation.

    Args:
        value (Any): Either a numeric seconds offset or a timedelta-style
            string.

    Returns:
        float: The offset in seconds as a float.

    Raises:
        ValueError: If ``value`` is a string that cannot be parsed.
    """
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    days = 0
    if "day" in text:
        day_part, _, text = text.partition(",")
        days = int(day_part.strip().split()[0])
        text = text.strip()
    parts = text.split(":")
    if len(parts) != 3:
        raise ValueError(f"Cannot parse timestamp: {value!r}")
    hours, minutes, seconds = (float(p) for p in parts)
    return days * 86400.0 + hours * 3600.0 + minutes * 60.0 + seconds


def transcript_segments_from_df(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a transcript DataFrame into segment dicts for the JSONL builder.

    The DataFrame is the sentence-merged transcript produced by
    :meth:`nextext.core.transcription.ExternalWhisperTranscriber.transcript_output`.
    The ``start`` / ``end`` columns are
    timedelta-style strings; ``speaker`` is optional. The emitted
    ``start_seconds`` / ``end_seconds`` values are integer-valued floats
    derived from those already-rounded strings — see
    :func:`parse_hhmmss_to_seconds` and :func:`format_hhmmss`.

    Args:
        df (pd.DataFrame): Transcript DataFrame with at least ``start``,
            ``end``, and ``text`` columns.

    Returns:
        list[dict[str, Any]]: Segment dicts with ``start_seconds``,
            ``end_seconds``, ``text``, and optional ``speaker``.
    """
    if df.empty or "text" not in df.columns:
        return []
    segments: list[dict[str, Any]] = []
    has_speaker = "speaker" in df.columns
    for _, row in df.iterrows():
        try:
            start_seconds = parse_hhmmss_to_seconds(row["start"])
            end_seconds = parse_hhmmss_to_seconds(row["end"])
        except (KeyError, ValueError, TypeError):
            continue
        segment: dict[str, Any] = {
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,
            "text": str(row["text"]),
        }
        if has_speaker:
            speaker = row.get("speaker")
            if speaker is not None and str(speaker).strip():
                segment["speaker"] = str(speaker)
        segments.append(segment)
    return segments


def _coerce_float(value: Any, *, field: str) -> float:
    """Convert a raw segment field to ``float`` with a helpful error.

    Args:
        value (Any): The raw value from a segment mapping.
        field (str): Name of the field being coerced, used in the error
            message when conversion fails.

    Returns:
        float: ``value`` as a Python float.

    Raises:
        TypeError: If ``value`` cannot be interpreted as a number.
    """
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Segment field '{field}' must be numeric, got {type(value).__name__}.") from exc


def _segment_to_record(
    segment: Mapping[str, Any],
    *,
    index: int,
    source_file: str,
    source_file_hash: str | None,
    language: str | None,
    task: str,
) -> dict[str, Any]:
    """Build the per-line JSON record for a single segment.

    The record shape is documented at module level. ``speaker`` and
    ``source_file_hash`` are omitted entirely — not set to ``None`` — when
    the caller did not provide them, so the docint ingest side can rely on
    key presence.

    Args:
        segment (Mapping[str, Any]): One row from the sentence-level
            transcript. Must contain ``start_seconds``, ``end_seconds`` and
            ``text``.
        index (int): Zero-based sentence index assigned by the caller.
        source_file (str): Original upload file name.
        source_file_hash (str | None): ``sha256:<hex>`` digest of the
            source audio, or ``None`` to omit the key.
        language (str | None): ISO 639-1 language code of the transcript
            text. May be ``None`` when unknown.
        task (str): Whisper task, ``"transcribe"`` or ``"translate"``.

    Returns:
        dict[str, Any]: The ordered record ready for ``json.dumps``.

    Raises:
        KeyError: If the segment is missing a required field.
        TypeError: If ``start_seconds`` or ``end_seconds`` are not numeric.
    """
    if "start_seconds" not in segment:
        raise KeyError("Segment is missing required field 'start_seconds'.")
    if "end_seconds" not in segment:
        raise KeyError("Segment is missing required field 'end_seconds'.")
    if "text" not in segment:
        raise KeyError("Segment is missing required field 'text'.")

    start_seconds = _coerce_float(segment["start_seconds"], field="start_seconds")
    end_seconds = _coerce_float(segment["end_seconds"], field="end_seconds")
    text = str(segment["text"])

    record: dict[str, Any] = {
        "source_file": source_file,
    }
    if source_file_hash is not None:
        record["source_file_hash"] = source_file_hash
    record["language"] = language
    record["task"] = task
    record["sentence_index"] = index
    record["start_seconds"] = start_seconds
    record["end_seconds"] = end_seconds
    record["start_ts"] = format_hhmmss(start_seconds)
    record["end_ts"] = format_hhmmss(end_seconds)

    speaker = segment.get("speaker")
    if speaker is not None:
        speaker_str = str(speaker).strip()
        if speaker_str:
            record["speaker"] = speaker_str

    record["text"] = text
    return record


def build_docint_jsonl(
    *,
    source_file: str,
    source_file_hash: str | None,
    language: str | None,
    task: str,
    segments: Sequence[Mapping[str, Any]],
) -> bytes:
    r"""Serialize a sentence-level transcript to docint-flavoured JSONL bytes.

    The payload is one JSON object per line, newline-terminated including
    the final line. Unicode text is preserved verbatim (no ASCII escaping)
    so downstream search indexing does not need to undo ``\\uXXXX``
    sequences. The ``sentence_index`` field is assigned 0, 1, 2, … in input
    order regardless of segment ordering supplied by the caller.

    Schema (one object per line):

    - ``source_file`` (str) — original upload file name.
    - ``source_file_hash`` (str, optional) — ``sha256:<hex>`` digest of the
      source audio. Omitted entirely when ``source_file_hash`` is ``None``.
    - ``language`` (str | None) — ISO 639-1 language code or ``None``.
    - ``task`` (str) — Whisper task, ``"transcribe"`` or ``"translate"``.
    - ``sentence_index`` (int) — 0-based line index.
    - ``start_seconds`` / ``end_seconds`` (float) — raw segment offsets.
    - ``start_ts`` / ``end_ts`` (str) — zero-padded ``HH:MM:SS`` strings.
    - ``speaker`` (str, optional) — diarization label. Omitted entirely
      when the segment has no speaker.
    - ``text`` (str) — sentence-level transcript text.

    Args:
        source_file (str): Original upload file name recorded in every
            record.
        source_file_hash (str | None): ``sha256:<hex>`` digest of the
            source audio bytes, or ``None`` to omit the key.
        language (str | None): ISO 639-1 language code attached to every
            record. ``None`` is preserved as JSON ``null`` — callers should
            pass ``None`` only when language truly could not be resolved.
        task (str): ``"transcribe"`` or ``"translate"``.
        segments (Sequence[Mapping[str, Any]]): Sentence-level rows with at
            least ``start_seconds`` (float), ``end_seconds`` (float), and
            ``text`` (str). ``speaker`` (str) is optional.

    Returns:
        bytes: UTF-8 encoded JSONL payload. Empty ``segments`` yields
            ``b""``.

    Raises:
        KeyError: If any segment is missing a required field.
        TypeError: If ``start_seconds`` or ``end_seconds`` are not numeric.
    """
    lines: list[str] = []
    for index, segment in enumerate(segments):
        record = _segment_to_record(
            segment,
            index=index,
            source_file=source_file,
            source_file_hash=source_file_hash,
            language=language,
            task=task,
        )
        lines.append(json.dumps(record, ensure_ascii=False))

    if not lines:
        return b""
    payload = "\n".join(lines) + "\n"
    return payload.encode("utf-8")
