"""Tests for the Streamlit application helpers."""

import pytest

from nextext.app import (
    _default_target_language,
    _download_file_name,
    _progress_value,
    _result_file_names,
    _select_result,
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
