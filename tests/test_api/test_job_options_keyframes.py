import pytest
from pydantic import ValidationError

from nextext.api.schemas import JobOptions


def test_job_options_accepts_keyframe_fields() -> None:
    opts = JobOptions.model_validate({"keyframes_per_minute": 6, "keyframes_max": 30})
    assert opts.keyframes_per_minute == 6
    assert opts.keyframes_max == 30


def test_job_options_keyframe_defaults() -> None:
    opts = JobOptions.model_validate({})
    assert opts.keyframes_per_minute == 4
    assert opts.keyframes_max == 20


def test_job_options_rejects_negative_rate() -> None:
    with pytest.raises(ValidationError):
        JobOptions.model_validate({"keyframes_per_minute": -1})
