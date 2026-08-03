"""Tests for the bundled ISO-code → English-language-name resolution.

``language_name_from_code`` replaces the former pycountry dependency (22 MB
of ISO databases for a single lookup pattern) with the already-bundled
``whisper_languages.json`` mapping.
"""

import pytest

from nextext.core.docint_transcript import language_name
from nextext.utils.mappings_loader import language_name_from_code


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("en", "English"),
        ("de", "German"),
        ("he", "Hebrew"),
        ("DE", "German"),
        ("zh-cn", "Chinese"),
        ("de-CH", "German"),
    ],
)
def test_language_name_from_code_resolves_known_codes(code: str, expected: str) -> None:
    """Known ISO codes resolve to their English names, case/locale-insensitively.

    Args:
        code (str): The input language code.
        expected (str): The expected English language name.
    """
    assert language_name_from_code(code) == expected


@pytest.mark.parametrize("code", [None, "", "xx", "not-a-code"])
def test_language_name_from_code_falls_back_to_default(code: str | None) -> None:
    """Empty or unknown codes return the caller-supplied default.

    Args:
        code (str | None): The unresolvable input code.
    """
    assert language_name_from_code(code) == ""
    assert language_name_from_code(code, default="fallback") == "fallback"


def test_docint_language_name_keeps_defaults() -> None:
    """The docint-transcript wrapper keeps its German default and code fallback."""
    assert language_name(None) == "German"
    assert language_name("fr") == "French"
    assert language_name("xx") == "xx"
