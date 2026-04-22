"""Tests for the Streamlit application helpers."""

import io
import json
import zipfile

import pandas as pd  # type: ignore[import-untyped]
import pytest

from nextext.app import (
    _build_docint_jsonl_archive,
    _default_target_language,
    _download_file_name,
    _parse_hhmmss_to_seconds,
    _progress_value,
    _result_file_names,
    _select_result,
    _transcript_segments_from_df,
)


def test_default_target_language_prefers_german_locale_code() -> None:
    """Test that the target language default tracks the German locale entry."""
    language_maps = {
        "en": "English",
        "de-DE": "German (Germany)",
        "fr-FR": "French (France)",
    }
    language_names = sorted(language_maps.values())

    default_index, default_code = _default_target_language(
        language_maps,
        language_names,
    )

    assert default_code == "de-DE"
    assert language_names[default_index] == "German (Germany)"


def test_default_target_language_falls_back_to_english() -> None:
    """Test that the target language default falls back to English."""
    language_maps = {
        "en": "English",
        "fr-FR": "French (France)",
    }
    language_names = sorted(language_maps.values())

    default_index, default_code = _default_target_language(
        language_maps,
        language_names,
    )

    assert default_code == "en"
    assert language_names[default_index] == "English"


def test_progress_value_tracks_file_stage_progress() -> None:
    """Test that multi-file progress reflects both file and stage position."""
    progress = _progress_value(
        file_index=2,
        total_files=4,
        stage_index=2,
        total_stages=4,
    )

    assert progress == 0.375


def test_result_file_names_prefers_uploaded_file_name() -> None:
    """Test that result labels use the stored file names."""
    result_names = _result_file_names(
        [
            {"file_name": "first.wav"},
            {"file_name": "second.mp3"},
        ]
    )

    assert result_names == ["first.wav", "second.mp3"]


def test_select_result_returns_requested_file() -> None:
    """Test that selecting a file returns its stored result entry."""
    results = [
        {"file_name": "first.wav", "summary": "one"},
        {"file_name": "second.mp3", "summary": "two"},
    ]

    selected = _select_result(results, "second.mp3")

    assert selected["summary"] == "two"


def test_select_result_falls_back_to_first_entry() -> None:
    """Test that result selection falls back when no file matches."""
    results = [
        {"file_name": "first.wav", "summary": "one"},
        {"file_name": "second.mp3", "summary": "two"},
    ]

    selected = _select_result(results, "missing.wav")

    assert selected["summary"] == "one"


def test_select_result_requires_non_empty_results() -> None:
    """Test that selecting a result requires at least one entry."""
    with pytest.raises(ValueError, match="At least one result"):
        _select_result([], None)


def test_download_file_name_uses_original_upload_stem() -> None:
    """Test that downloads keep the original uploaded filename stem."""
    download_name = _download_file_name(
        {"file_name": "meeting-recording.m4a"},
        "summary.txt",
    )

    assert download_name == "meeting-recording_summary.txt"


def test_parse_hhmmss_to_seconds_handles_timedelta_strings() -> None:
    """Test that timedelta-style strings round-trip to seconds floats."""
    assert _parse_hhmmss_to_seconds("0:00:12") == 12.0
    assert _parse_hhmmss_to_seconds("1:02:05") == 3725.0
    assert _parse_hhmmss_to_seconds("10:00:00") == 36000.0
    assert _parse_hhmmss_to_seconds("1 day, 0:00:00") == 86400.0
    assert _parse_hhmmss_to_seconds(42) == 42.0


def test_transcript_segments_from_df_round_trips_speaker_column() -> None:
    """Test that a transcript DataFrame maps to segment dicts correctly."""
    df = pd.DataFrame(
        [
            {"start": "0:00:00", "end": "0:00:04", "speaker": "S1", "text": "hi"},
            {"start": "0:00:04", "end": "0:00:09", "speaker": "", "text": "there"},
        ]
    )
    segments = _transcript_segments_from_df(df)
    assert segments[0]["start_seconds"] == 0.0
    assert segments[0]["end_seconds"] == 4.0
    assert segments[0]["speaker"] == "S1"
    # Empty speaker string is filtered out entirely.
    assert "speaker" not in segments[1]


def test_build_docint_jsonl_archive_single_file_returns_plain_jsonl() -> None:
    """Test that a single processed file yields a plain JSONL payload."""
    results = [
        {
            "file_name": "interview.mp3",
            "source_file_hash": "sha256:abc",
            "transcript_language": "de",
            "task": "transcribe",
            "transcript": pd.DataFrame(
                [
                    {
                        "start": "0:00:00",
                        "end": "0:00:04",
                        "speaker": "S1",
                        "text": "Hallo.",
                    }
                ]
            ),
        }
    ]
    data, file_name, mime = _build_docint_jsonl_archive(results, "stem")
    assert mime == "application/x-ndjson"
    assert file_name == "interview.docint.jsonl"
    record = json.loads(data.decode("utf-8").splitlines()[0])
    assert record["source_file"] == "interview.mp3"
    assert record["source_file_hash"] == "sha256:abc"
    assert record["language"] == "de"
    assert record["speaker"] == "S1"


def test_build_docint_jsonl_archive_multi_file_returns_zip() -> None:
    """Test that multiple processed files are zipped together."""
    results = [
        {
            "file_name": "a.mp3",
            "source_file_hash": None,
            "transcript_language": "en",
            "task": "transcribe",
            "transcript": pd.DataFrame(
                [{"start": "0:00:00", "end": "0:00:01", "text": "one"}]
            ),
        },
        {
            "file_name": "b.mp3",
            "source_file_hash": None,
            "transcript_language": "en",
            "task": "transcribe",
            "transcript": pd.DataFrame(
                [{"start": "0:00:00", "end": "0:00:01", "text": "two"}]
            ),
        },
    ]
    data, file_name, mime = _build_docint_jsonl_archive(results, "bundle")
    assert mime == "application/zip"
    assert file_name == "bundle_docint.zip"
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = sorted(zf.namelist())
        assert names == ["bundle/a.jsonl", "bundle/b.jsonl"]


def test_build_docint_jsonl_archive_skips_empty_transcripts() -> None:
    """Test that results with empty transcripts yield empty output."""
    results = [
        {
            "file_name": "empty.mp3",
            "transcript": pd.DataFrame(columns=["start", "end", "text"]),
        }
    ]
    data, file_name, mime = _build_docint_jsonl_archive(results, "stem")
    assert data == b""
    assert file_name == "stem.jsonl"
    assert mime == "application/x-ndjson"
