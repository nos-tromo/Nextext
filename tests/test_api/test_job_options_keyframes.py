"""Tests for the keyframe fields on ``JobOptions`` (nextext.api.schemas)."""

import pytest
from pydantic import ValidationError

from nextext.api.schemas import JobOptions


def _clear_keyframe_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Removes both keyframe env vars so defaults resolve to the hardcoded fallback.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for patching
            environment variables.
    """
    monkeypatch.delenv("KEYFRAMES_PER_MINUTE", raising=False)
    monkeypatch.delenv("KEYFRAMES_MAX", raising=False)


def test_job_options_accepts_keyframe_fields() -> None:
    """Explicit keyframe fields round-trip onto the parsed options."""
    opts = JobOptions.model_validate({"keyframes_per_minute": 6, "keyframes_max": 30})
    assert opts.keyframes_per_minute == 6
    assert opts.keyframes_max == 30


def test_job_options_keyframe_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Omitting the keyframe fields with no operator env falls back to 4 / 20.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for patching
            environment variables.
    """
    _clear_keyframe_env(monkeypatch)

    opts = JobOptions.model_validate({})

    assert opts.keyframes_per_minute == 4
    assert opts.keyframes_max == 20


def test_job_options_keyframe_defaults_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """An operator-configured env overrides the hardcoded keyframe defaults.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for patching
            environment variables.
    """
    monkeypatch.setenv("KEYFRAMES_PER_MINUTE", "6")
    monkeypatch.setenv("KEYFRAMES_MAX", "30")

    opts = JobOptions()

    assert opts.keyframes_per_minute == 6
    assert opts.keyframes_max == 30


def test_job_options_keyframe_max_env_clamps_to_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    """A KEYFRAMES_MAX above the hard cap clamps the default to 200 rather than erroring.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for patching
            environment variables.
    """
    monkeypatch.delenv("KEYFRAMES_PER_MINUTE", raising=False)
    monkeypatch.setenv("KEYFRAMES_MAX", "999")

    opts = JobOptions()

    assert opts.keyframes_max == 200


def test_job_options_keyframe_rate_env_clamps_to_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """A negative KEYFRAMES_PER_MINUTE clamps the default to 0 rather than erroring.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for patching
            environment variables.
    """
    monkeypatch.setenv("KEYFRAMES_PER_MINUTE", "-5")
    monkeypatch.delenv("KEYFRAMES_MAX", raising=False)

    opts = JobOptions()

    assert opts.keyframes_per_minute == 0


def test_job_options_explicit_value_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit per-request value still wins over an operator-configured env default.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for patching
            environment variables.
    """
    monkeypatch.setenv("KEYFRAMES_PER_MINUTE", "6")

    opts = JobOptions.model_validate({"keyframes_per_minute": 10})

    assert opts.keyframes_per_minute == 10


def test_job_options_rejects_negative_rate() -> None:
    """A negative ``keyframes_per_minute`` violates the ``ge=0`` bound."""
    with pytest.raises(ValidationError):
        JobOptions.model_validate({"keyframes_per_minute": -1})


def test_job_options_rejects_max_above_ceiling() -> None:
    """A ``keyframes_max`` above the ``le=200`` ceiling is rejected."""
    with pytest.raises(ValidationError):
        JobOptions.model_validate({"keyframes_max": 201})
