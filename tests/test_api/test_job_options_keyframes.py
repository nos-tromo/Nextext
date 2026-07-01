"""Tests for the keyframe fields on ``JobOptions`` (nextext.api.schemas)."""

import pytest
from pydantic import ValidationError

from nextext.api.schemas import JobOptions


def test_job_options_accepts_keyframe_fields() -> None:
    """Explicit keyframe fields round-trip onto the parsed options."""
    opts = JobOptions.model_validate({"keyframes_per_minute": 6, "keyframes_max": 30})
    assert opts.keyframes_per_minute == 6
    assert opts.keyframes_max == 30


def test_job_options_keyframe_defaults() -> None:
    """Omitting the keyframe fields falls back to the documented defaults."""
    opts = JobOptions.model_validate({})
    assert opts.keyframes_per_minute == 4
    assert opts.keyframes_max == 20


def test_job_options_rejects_negative_rate() -> None:
    """A negative ``keyframes_per_minute`` violates the ``ge=0`` bound."""
    with pytest.raises(ValidationError):
        JobOptions.model_validate({"keyframes_per_minute": -1})


def test_job_options_rejects_max_above_ceiling() -> None:
    """A ``keyframes_max`` above the ``le=200`` ceiling is rejected."""
    with pytest.raises(ValidationError):
        JobOptions.model_validate({"keyframes_max": 201})
