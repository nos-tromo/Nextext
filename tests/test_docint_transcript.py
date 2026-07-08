"""Tests for the docint JSONL transcript exporter."""

from __future__ import annotations

import json
from typing import Any

import pytest

from nextext.core.docint_transcript import build_docint_jsonl, format_hhmmss


def _decode_lines(payload: bytes) -> list[dict[str, Any]]:
    """Decode a JSONL payload into a list of dictionaries.

    Args:
        payload (bytes): UTF-8 encoded JSONL bytes.

    Returns:
        list[dict]: Parsed records, one per line.
    """
    assert payload.endswith(b"\n"), "JSONL payload must be newline-terminated."
    text = payload.decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line]


def test_build_docint_jsonl_golden() -> None:
    """Test the full serialization across the common segment shapes."""
    segments = [
        {
            "start_seconds": 0.0,
            "end_seconds": 4.2,
            "text": "Guten Tag — heute geht es um Äpfel.",
            "speaker": "SPEAKER_00",
        },
        {
            "start_seconds": 4.2,
            "end_seconds": 9.75,
            "text": "Hello 中国, this segment has no speaker.",
        },
        {
            "start_seconds": 9.75,
            "end_seconds": 12.0,
            "text": "Final line.",
            "speaker": "SPEAKER_01",
        },
    ]
    payload = build_docint_jsonl(
        source_file="interview.mp3",
        source_file_hash="sha256:abc123",
        language="de",
        segments=segments,
    )

    # No ASCII escaping — unicode is preserved verbatim.
    assert "Äpfel" in payload.decode("utf-8")
    assert "中国" in payload.decode("utf-8")
    assert b"\\u" not in payload

    records = _decode_lines(payload)
    assert len(records) == 3

    first, second, third = records
    assert first == {
        "source_file": "interview.mp3",
        "source_file_hash": "sha256:abc123",
        "language": "de",
        "sentence_index": 0,
        "start_seconds": 0.0,
        "end_seconds": 4.2,
        "start_ts": "00:00:00",
        "end_ts": "00:00:04",
        "speaker": "SPEAKER_00",
        "text": "Guten Tag — heute geht es um Äpfel.",
    }
    assert "speaker" not in second
    assert second["sentence_index"] == 1
    assert second["start_ts"] == "00:00:04"
    assert second["end_ts"] == "00:00:09"
    assert third["speaker"] == "SPEAKER_01"
    assert third["sentence_index"] == 2


def test_build_docint_jsonl_includes_language() -> None:
    """Test that language is emitted on every record."""
    payload = build_docint_jsonl(
        source_file="x.wav",
        source_file_hash=None,
        language="de",
        segments=[
            {"start_seconds": 0.0, "end_seconds": 1.0, "text": "a"},
            {"start_seconds": 1.0, "end_seconds": 2.0, "text": "b"},
        ],
    )
    records = _decode_lines(payload)
    assert [r["language"] for r in records] == ["de", "de"]


def test_build_docint_jsonl_omits_speaker_when_absent() -> None:
    """Test that the speaker key is absent rather than null."""
    payload = build_docint_jsonl(
        source_file="short.wav",
        source_file_hash=None,
        language="en",
        segments=[
            {"start_seconds": 0.0, "end_seconds": 1.0, "text": "No speaker."},
            {
                "start_seconds": 1.0,
                "end_seconds": 2.0,
                "text": "Empty speaker string.",
                "speaker": "   ",
            },
            {
                "start_seconds": 2.0,
                "end_seconds": 3.0,
                "text": "Explicit None.",
                "speaker": None,
            },
        ],
    )
    records = _decode_lines(payload)
    for record in records:
        assert "speaker" not in record


def test_build_docint_jsonl_omits_file_hash_when_none() -> None:
    """Test that the source_file_hash key is absent rather than null."""
    payload = build_docint_jsonl(
        source_file="short.wav",
        source_file_hash=None,
        language="en",
        segments=[
            {"start_seconds": 0.0, "end_seconds": 1.0, "text": "."},
        ],
    )
    record = _decode_lines(payload)[0]
    assert "source_file_hash" not in record


def test_build_docint_jsonl_sentence_index_is_monotonic() -> None:
    """Test that sentence_index is assigned 0..N-1 in input order."""
    segments = [
        {"start_seconds": 30.0, "end_seconds": 32.0, "text": "c"},
        {"start_seconds": 0.0, "end_seconds": 1.0, "text": "a"},
        {"start_seconds": 10.0, "end_seconds": 12.0, "text": "b"},
        {"start_seconds": 100.0, "end_seconds": 101.0, "text": "d"},
    ]
    payload = build_docint_jsonl(
        source_file="ordered.wav",
        source_file_hash=None,
        language="en",
        segments=segments,
    )
    records = _decode_lines(payload)
    assert [r["sentence_index"] for r in records] == [0, 1, 2, 3]
    assert [r["text"] for r in records] == ["c", "a", "b", "d"]


def test_build_docint_jsonl_formats_timestamps() -> None:
    """Test that HH:MM:SS timestamps handle hour-scale offsets."""
    payload = build_docint_jsonl(
        source_file="long.wav",
        source_file_hash=None,
        language="en",
        segments=[
            {
                "start_seconds": 3725.5,
                "end_seconds": 3730.0,
                "text": "hour mark",
            },
        ],
    )
    record = _decode_lines(payload)[0]
    assert record["start_ts"] == "01:02:05"
    assert record["end_ts"] == "01:02:10"


def test_build_docint_jsonl_truncates_fractional_seconds() -> None:
    """Test that fractional seconds are truncated for the HH:MM:SS label."""
    payload = build_docint_jsonl(
        source_file="trunc.wav",
        source_file_hash=None,
        language="en",
        segments=[
            {"start_seconds": 0.9, "end_seconds": 1.9, "text": "round"},
        ],
    )
    record = _decode_lines(payload)[0]
    assert record["start_ts"] == "00:00:00"
    assert record["end_ts"] == "00:00:01"


def test_build_docint_jsonl_null_language_serializes_as_json_null() -> None:
    """Test that ``language=None`` is preserved as a JSON ``null`` field."""
    payload = build_docint_jsonl(
        source_file="x.wav",
        source_file_hash=None,
        language=None,
        segments=[{"start_seconds": 0.0, "end_seconds": 1.0, "text": "."}],
    )
    record = _decode_lines(payload)[0]
    assert "language" in record and record["language"] is None


def test_build_docint_jsonl_empty_segments_returns_empty_bytes() -> None:
    """Test that an empty segment list yields an empty payload."""
    payload = build_docint_jsonl(
        source_file="empty.wav",
        source_file_hash=None,
        language=None,
        segments=[],
    )
    assert payload == b""


def test_build_docint_jsonl_final_line_newline_terminated() -> None:
    """Test that the payload ends in a newline even with a single segment."""
    payload = build_docint_jsonl(
        source_file="one.wav",
        source_file_hash=None,
        language="en",
        segments=[{"start_seconds": 0.0, "end_seconds": 1.0, "text": "."}],
    )
    assert payload.endswith(b"\n")
    assert payload.count(b"\n") == 1


def test_build_docint_jsonl_missing_required_field_raises() -> None:
    """Test that a missing required field surfaces as a KeyError."""
    with pytest.raises(KeyError, match="start_seconds"):
        build_docint_jsonl(
            source_file="bad.wav",
            source_file_hash=None,
            language="en",
            segments=[{"end_seconds": 1.0, "text": "missing start"}],
        )


def test_format_hhmmss_rejects_negative() -> None:
    """Test that format_hhmmss rejects negative offsets."""
    with pytest.raises(ValueError, match="non-negative"):
        format_hhmmss(-1.0)


def test_format_hhmmss_truncates_fractional_seconds() -> None:
    """Test that fractional seconds are truncated rather than rounded."""
    assert format_hhmmss(0.4) == "00:00:00"
    assert format_hhmmss(0.9) == "00:00:00"
    assert format_hhmmss(59.9) == "00:00:59"
    assert format_hhmmss(3600.0) == "01:00:00"
