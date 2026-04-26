"""Tests for the hate speech detection agent."""

from typing import Any, cast

import pytest

from nextext.core.hate_speech import (
    HateSpeechDetector,
    _default_detection,
    _normalize_confidence,
    _parse_hate_speech_payload,
)


# ---------------------------------------------------------------------------
# _parse_hate_speech_payload
# ---------------------------------------------------------------------------


def test_parse_valid_json_hate_speech_true() -> None:
    """Test parsing a well-formed JSON response that flags hate speech."""
    raw = '{"hate_speech": true, "category": "racism", "confidence": "high", "reason": "Contains slurs"}'
    result = _parse_hate_speech_payload(raw)

    assert result["hate_speech"] is True
    assert result["category"] == "racism"
    assert result["confidence"] == "high"
    assert result["reason"] == "Contains slurs"


def test_parse_valid_json_no_hate_speech() -> None:
    """Test parsing a well-formed JSON response that does not flag hate speech."""
    raw = (
        '{"hate_speech": false, "category": "none", "confidence": "low", "reason": ""}'
    )
    result = _parse_hate_speech_payload(raw)

    assert result["hate_speech"] is False
    assert result["category"] == "none"


def test_parse_json_embedded_in_prose() -> None:
    """Test that JSON embedded inside prose is extracted correctly."""
    raw = 'Sure! Here is my answer: {"hate_speech": true, "category": "sexism", "confidence": "medium", "reason": "Derogatory"} Hope that helps.'
    result = _parse_hate_speech_payload(raw)

    assert result["hate_speech"] is True
    assert result["category"] == "sexism"
    assert result["confidence"] == "medium"


def test_parse_malformed_json_returns_default() -> None:
    """Test that malformed JSON returns the default safe detection result."""
    result = _parse_hate_speech_payload("{not valid json}")

    assert result["hate_speech"] is False
    assert result["category"] == "none"
    assert result["confidence"] == "low"


def test_parse_garbage_input_returns_default() -> None:
    """Test that completely unparseable input returns the default safe detection result."""
    result = _parse_hate_speech_payload("I cannot answer this request.")

    assert result["hate_speech"] is False
    assert result["reason"] == "Parse error"


def test_parse_normalises_unknown_confidence() -> None:
    """Test that an unrecognised confidence value is normalised to 'low'."""
    raw = '{"hate_speech": false, "category": "none", "confidence": "unknown", "reason": ""}'
    result = _parse_hate_speech_payload(raw)

    assert result["confidence"] == "low"


def test_parse_category_is_lowercased() -> None:
    """Test that category strings are lowercased during parsing."""
    raw = '{"hate_speech": true, "category": "Racism", "confidence": "High", "reason": "test"}'
    result = _parse_hate_speech_payload(raw)

    assert result["category"] == "racism"
    assert result["confidence"] == "high"


# ---------------------------------------------------------------------------
# _normalize_confidence
# ---------------------------------------------------------------------------


def test_normalize_confidence_accepts_valid_values() -> None:
    """Test that valid confidence strings are returned unchanged."""
    assert _normalize_confidence("high") == "high"
    assert _normalize_confidence("medium") == "medium"
    assert _normalize_confidence("low") == "low"


def test_normalize_confidence_handles_mixed_case() -> None:
    """Test that mixed-case confidence strings are normalised."""
    assert _normalize_confidence("High") == "high"
    assert _normalize_confidence("MEDIUM") == "medium"


def test_normalize_confidence_fallback_to_low() -> None:
    """Test that unrecognised confidence values fall back to 'low'."""
    assert _normalize_confidence("very high") == "low"
    assert _normalize_confidence("") == "low"


# ---------------------------------------------------------------------------
# _default_detection
# ---------------------------------------------------------------------------


def test_default_detection_is_safe() -> None:
    """Test that the default detection result is always safe (hate_speech=False)."""
    result = _default_detection()

    assert result["hate_speech"] is False
    assert result["category"] == "none"
    assert result["confidence"] == "low"
    assert result["reason"] == "Parse error"


# ---------------------------------------------------------------------------
# HateSpeechDetector
# ---------------------------------------------------------------------------


def test_detector_calls_inference_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that HateSpeechDetector.detect() calls the inference pipeline and parses the result.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture for patching.
    """
    from nextext.core.openai_cfg import InferencePipeline

    class DummyPipeline:
        """Structural stand-in for ``InferencePipeline`` used by the detector."""

        def load_prompt(self, keyword: str = "system") -> str:
            """Return a stub prompt template for the hate-speech keyword.

            Args:
                keyword (str): The prompt keyword requested by the caller.

            Returns:
                str: A minimal template containing a ``{text}`` placeholder.
            """
            assert keyword == "hate_speech"
            return "Analyze: {text}"

        def call_model(self, prompt: str, **kwargs: Any) -> str:
            """Return a canned JSON payload and assert the prompt content.

            Args:
                prompt (str): The formatted prompt passed by the detector.
                **kwargs (Any): Remaining ``InferencePipeline.call_model``
                    keyword arguments, ignored by this stub.

            Returns:
                str: A JSON string describing a clean (non-hate) result.
            """
            assert "hello world" in prompt
            return '{"hate_speech": false, "category": "none", "confidence": "low", "reason": "clean"}'

    detector = HateSpeechDetector(
        cast(InferencePipeline, DummyPipeline()), max_chars=2048
    )
    result = detector.detect("hello world")

    assert result["hate_speech"] is False
    assert result["reason"] == "clean"


def test_detector_truncates_text_to_max_chars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that HateSpeechDetector.detect() truncates text to max_chars.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture for patching.
    """
    from nextext.core.openai_cfg import InferencePipeline

    received_prompts: list[str] = []

    class DummyPipeline:
        """Structural stand-in for ``InferencePipeline`` that records prompts."""

        def load_prompt(self, keyword: str = "system") -> str:
            """Return a passthrough template exposing the raw text argument.

            Args:
                keyword (str): The prompt keyword requested by the caller.

            Returns:
                str: A template consisting only of the ``{text}`` placeholder.
            """
            return "{text}"

        def call_model(self, prompt: str, **kwargs: Any) -> str:
            """Capture the prompt and return a canned clean payload.

            Args:
                prompt (str): The formatted prompt passed by the detector.
                **kwargs (Any): Remaining ``InferencePipeline.call_model``
                    keyword arguments, ignored by this stub.

            Returns:
                str: A JSON string describing a clean (non-hate) result.
            """
            received_prompts.append(prompt)
            return '{"hate_speech": false, "category": "none", "confidence": "low", "reason": ""}'

    detector = HateSpeechDetector(cast(InferencePipeline, DummyPipeline()), max_chars=5)
    detector.detect("hello world")

    assert received_prompts[0] == "hello"
