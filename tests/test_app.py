"""Tests for the Streamlit application helpers."""

from nextext.app import _default_target_language


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
