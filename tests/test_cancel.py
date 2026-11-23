"""Tests for cooperative job cancellation (``nextext.core.cancel``).

A deleted job's pipeline thread stops at its next check point by way of
``JobCancelled``. These tests pin the module's own contract: ``check`` does
nothing without a bound event or while it is unset, the binding stays inside its
scope, and the exception passes through the pipeline's fail-soft guards.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from nextext.core.cancel import JobCancelled, bind_cancel, check
from nextext.core.inference_trace import record_inference, trace_inference
from nextext.core.visual_context import describe_keyframes


def test_check_is_a_noop_without_a_bound_event() -> None:
    """Outside any job — the CLI, a plain call — ``check`` raises nothing."""
    check()


def test_check_is_a_noop_while_the_bound_event_is_unset() -> None:
    """A job nobody cancelled runs through its check points."""
    with bind_cancel(threading.Event()):
        check()


def test_check_raises_once_the_bound_event_is_set() -> None:
    """The first check point after the event is set stops the job."""
    event = threading.Event()
    with bind_cancel(event):
        check()
        event.set()
        with pytest.raises(JobCancelled):
            check()


def test_binding_does_not_leak_out_of_its_scope() -> None:
    """After the block, a set event no longer reaches ``check``."""
    event = threading.Event()
    event.set()
    with bind_cancel(event):
        pass
    check()


def test_binding_is_restored_after_a_nested_scope() -> None:
    """An inner binding gives way to the outer one when it ends."""
    outer = threading.Event()
    inner = threading.Event()
    inner.set()
    with bind_cancel(outer):
        with bind_cancel(inner), pytest.raises(JobCancelled):
            check()
        check()


def test_binding_is_released_when_the_block_raises() -> None:
    """A block that fails does not leave its event bound."""
    event = threading.Event()
    event.set()
    with pytest.raises(ValueError, match="boom"):
        with bind_cancel(event):
            raise ValueError("boom")
    check()


def test_binding_does_not_cross_into_another_thread() -> None:
    """A thread started outside the scope sees no event; each job binds its own."""
    event = threading.Event()
    event.set()
    outcome: list[str] = []

    def other() -> None:
        try:
            check()
        except JobCancelled:
            outcome.append("cancelled")
        else:
            outcome.append("running")

    with bind_cancel(event):
        thread = threading.Thread(target=other)
        thread.start()
        thread.join()
    assert outcome == ["running"]


def test_job_cancelled_is_not_an_exception() -> None:
    """``except Exception`` cannot catch it: that is what makes it pass fail-soft guards."""
    assert issubclass(JobCancelled, BaseException)
    assert not issubclass(JobCancelled, Exception)


def test_a_fail_soft_guard_does_not_swallow_a_cancelled_job() -> None:
    """A stage that degrades on ``Exception`` still lets the cancellation through."""
    event = threading.Event()
    event.set()
    degraded: list[str] = []

    def fail_soft_stage() -> None:
        try:
            check()
        except Exception:
            degraded.append("swallowed")

    with bind_cancel(event), pytest.raises(JobCancelled):
        fail_soft_stage()
    assert degraded == []


class _CaptioningClient:
    """A stand-in chat client whose every request is reported to the trace."""

    def __init__(self) -> None:
        self.requests = 0

    def load_prompt(self, name: str) -> str:
        """Return a fixed prompt.

        Args:
            name: Ignored.

        Returns:
            str: The prompt.
        """
        return "Describe the frame."

    def call_vision(self, **kwargs: Any) -> str:
        """Answer one request and report it, as the real client does.

        Args:
            **kwargs: Ignored.

        Returns:
            str: A caption.
        """
        self.requests += 1
        record_inference("chat", "chat-test", ok=True)
        return "A grey square."


def test_the_captioning_guard_lets_a_cancelled_job_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real fail-soft guard of the caption loop does not swallow the cancellation.

    The request that returned is reported to the trace first; the loop stops
    before the next one.
    """
    from nextext.core import visual_context
    from nextext.core.keyframes import Keyframe

    monkeypatch.setattr(visual_context, "prepare_frame", lambda jpeg, *, max_side: jpeg)
    event = threading.Event()
    client = _CaptioningClient()
    samples = [Keyframe(time_sec=float(second), jpeg=b"x") for second in range(3)]

    with trace_inference() as trace, bind_cancel(event):
        event.set()
        with pytest.raises(JobCancelled):
            describe_keyframes(samples, client, max_frames=3, max_side=64)  # type: ignore[arg-type]

    assert client.requests == 1
    assert [tally.requests for tally in trace.tallies] == [1]
