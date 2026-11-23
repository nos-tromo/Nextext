"""Cooperative cancellation of a running job's pipeline thread.

A pipeline runs on one worker thread and cannot be interrupted from outside, so
a job that its owner deleted would otherwise run to its end and hold its worker
slot the whole time. The job manager binds the job's cancel event for the
duration of the pipeline (:func:`bind_cancel`); the places where the pipeline
passes control — a stage boundary, an inference request, a loop over packets or
frames — call :func:`check`, which raises :class:`JobCancelled` once the event
is set. A call already in flight (one upload to Whisper, one ``/diarize``
request) is not interrupted: the work stops at the next check point after it
returns.

The event lives in a context variable, so a caller that binds none — the CLI,
a test, any code outside a job — is unaffected: :func:`check` then does nothing.
``asyncio.to_thread`` copies the context into the thread, and every thread binds
its own job's event, so concurrent jobs never see each other's.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

__all__ = ["JobCancelled", "bind_cancel", "check"]


class JobCancelled(BaseException):
    """Raised at a check point once the job's cancel event is set.

    A ``BaseException`` on purpose, like ``KeyboardInterrupt``: the pipeline's
    fail-soft guards catch ``Exception`` to degrade a stage instead of failing
    the job, and a cancelled job must pass through every one of them. The job
    manager catches it on the pipeline thread, so it never reaches the event
    loop.
    """


_CANCEL: ContextVar[threading.Event | None] = ContextVar("nextext_job_cancel", default=None)


@contextmanager
def bind_cancel(event: threading.Event) -> Iterator[None]:
    """Make ``event`` the cancel event that :func:`check` consults inside the block.

    Args:
        event: Set when the job is to stop.

    Yields:
        None: Control, with the event bound.
    """
    token = _CANCEL.set(event)
    try:
        yield
    finally:
        _CANCEL.reset(token)


def check() -> None:
    """Stop the current job if its cancel event is set.

    Does nothing outside :func:`bind_cancel`, or while the bound event is unset.

    Raises:
        JobCancelled: If the bound cancel event is set.
    """
    event = _CANCEL.get()
    if event is not None and event.is_set():
        raise JobCancelled
