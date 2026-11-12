"""Per-stage record of the inference requests that carry job content.

Every client that sends job content to an inference endpoint — Whisper,
``/vad``, ``/diarize``, ``/gliner`` and the chat model — reports each request
here: which task and model it was for, whether it succeeded, and the token
counts its reply carried. A caller that wants the record opens
:func:`trace_inference` around a pipeline stage and reads the tallies when the
stage ends; the data-protection audit log turns them into one ``ai_call`` event
per task (:mod:`nextext.audit`). Outside a trace, reports are dropped.

The module also carries the correlation id of the request that started the
work (:func:`correlate`). The clients forward it as ``X-Request-ID``
(:func:`correlation_headers`), so the router's logs can be matched with the
audit trail.

Chat calls also carry the job owner's cache scope (:func:`cache_scope`,
:func:`cache_scope_headers`): an HMAC of the owner id under a key drawn when
the process starts, sent as ``X-Cache-Scope``. The router gives each scope its
own share of the chat model's prefix cache, so one owner cannot time another
owner's cached prompts. The key never leaves the process, so nobody outside it
can map a scope back to an owner, and a restart draws new scopes. Nothing else
about the caller — no identity, no purpose — leaves with a request.

All three are context variables: ``asyncio.to_thread`` copies them into the
thread that runs a job's pipeline, and every job's worker task has a context
of its own, so concurrent jobs never share a trace, a correlation id or a
scope.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "InferenceTrace",
    "TaskTally",
    "cache_scope",
    "cache_scope_headers",
    "correlate",
    "correlation_headers",
    "record_inference",
    "reply_usage",
    "trace_inference",
]

REQUEST_ID_HEADER = "X-Request-ID"
CACHE_SCOPE_HEADER = "X-Cache-Scope"
_CACHE_SCOPE_KEY = secrets.token_bytes(32)
_CACHE_SCOPE_LABEL = b"cache-scope/v1\0"


@dataclass
class TaskTally:
    """What one inference task of a stage sent and got back.

    Attributes:
        task: The inference task, e.g. ``asr``, ``diarize``, ``ner``, ``vad``
            or ``chat``.
        models: The model names requested, in first-use order.
        requests: Requests sent (or attempted), successful or not.
        failed: Requests that raised or returned an error.
        input_tokens: Prompt tokens summed over the replies that reported them.
        output_tokens: Completion tokens summed over the replies that reported
            them.
        usage_reported: Successful replies that reported token usage.
        served_models: The model names the endpoint reported serving, in
            first-seen order.
    """

    task: str
    models: list[str] = field(default_factory=list)
    requests: int = 0
    failed: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    usage_reported: int = 0
    served_models: list[str] = field(default_factory=list)

    @property
    def succeeded(self) -> int:
        """Requests that returned a usable reply.

        Returns:
            int: ``requests - failed``.
        """
        return self.requests - self.failed

    @property
    def model(self) -> str:
        """The requested model, or the comma-joined models if several were used.

        Returns:
            str: The model name(s).
        """
        return ",".join(self.models)

    @property
    def served_model(self) -> str | None:
        """The model the endpoint reported serving, when it reported one.

        Returns:
            str | None: The comma-joined served model name(s), or ``None``.
        """
        return ",".join(self.served_models) or None

    @property
    def tokens(self) -> tuple[int | None, int | None]:
        """Token totals, when every successful reply reported them.

        A partial sum would understate what was sent, so the totals are given
        only when the count is complete.

        Returns:
            tuple[int | None, int | None]: ``(input_tokens, output_tokens)``,
                or ``(None, None)``.
        """
        if self.succeeded > 0 and self.usage_reported == self.succeeded:
            return self.input_tokens, self.output_tokens
        return None, None


class InferenceTrace:
    """The inference requests made while a trace was open, tallied per task."""

    def __init__(self) -> None:
        """Start an empty trace."""
        self._lock = threading.Lock()
        self._tallies: dict[str, TaskTally] = {}

    def record(
        self,
        task: str,
        model: str,
        *,
        ok: bool,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        served_model: str | None = None,
    ) -> None:
        """Count one request.

        Args:
            task: The inference task.
            model: The model name the request asked for.
            ok: Whether the request returned a usable reply.
            input_tokens: Prompt tokens the reply reported.
            output_tokens: Completion tokens the reply reported.
            served_model: The model the endpoint reported serving.
        """
        with self._lock:
            tally = self._tallies.setdefault(task, TaskTally(task=task))
            tally.requests += 1
            if model and model not in tally.models:
                tally.models.append(model)
            if not ok:
                tally.failed += 1
                return
            if input_tokens is not None and output_tokens is not None:
                tally.input_tokens += input_tokens
                tally.output_tokens += output_tokens
                tally.usage_reported += 1
            if served_model and served_model not in tally.served_models:
                tally.served_models.append(served_model)

    @property
    def tallies(self) -> list[TaskTally]:
        """The tallies, in the order each task was first used.

        Returns:
            list[TaskTally]: One tally per task.
        """
        with self._lock:
            return list(self._tallies.values())


_TRACE: ContextVar[InferenceTrace | None] = ContextVar("nextext_inference_trace", default=None)
_REQUEST_ID: ContextVar[str | None] = ContextVar("nextext_request_id", default=None)
_CACHE_SCOPE: ContextVar[str | None] = ContextVar("nextext_cache_scope", default=None)


def record_inference(
    task: str,
    model: str,
    *,
    ok: bool,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    served_model: str | None = None,
) -> None:
    """Report one inference request to the open trace, if there is one.

    Args:
        task: The inference task, e.g. ``asr`` or ``chat``.
        model: The model name the request asked for.
        ok: Whether the request returned a usable reply.
        input_tokens: Prompt tokens the reply reported.
        output_tokens: Completion tokens the reply reported.
        served_model: The model the endpoint reported serving.
    """
    trace = _TRACE.get()
    if trace is not None:
        trace.record(
            task,
            model,
            ok=ok,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            served_model=served_model,
        )


@contextmanager
def trace_inference() -> Iterator[InferenceTrace]:
    """Collect the inference requests made inside the block.

    Yields:
        InferenceTrace: The trace, filled as requests are made.
    """
    trace = InferenceTrace()
    token = _TRACE.set(trace)
    try:
        yield trace
    finally:
        _TRACE.reset(token)


@contextmanager
def correlate(request_id: str | None) -> Iterator[None]:
    """Forward ``request_id`` on the inference requests made inside the block.

    Args:
        request_id: The correlation id of the request that started the work,
            already validated; ``None`` forwards nothing.

    Yields:
        None: Control, with the correlation id in effect.
    """
    token = _REQUEST_ID.set(request_id)
    try:
        yield
    finally:
        _REQUEST_ID.reset(token)


def correlation_headers() -> dict[str, str]:
    """Return the headers an inference request carries for correlation.

    Returns:
        dict[str, str]: ``{"X-Request-ID": <id>}`` inside :func:`correlate`
            with an id, otherwise empty.
    """
    request_id = _REQUEST_ID.get()
    return {REQUEST_ID_HEADER: request_id} if request_id else {}


@contextmanager
def cache_scope(owner_id: str | None) -> Iterator[None]:
    """Send ``owner_id``'s cache scope on the chat calls made inside the block.

    Args:
        owner_id: The owner the work runs for; ``None`` or empty binds no
            scope.

    Yields:
        None: Control, with the scope in effect.
    """
    scope = None
    if owner_id:
        scope = hmac.new(_CACHE_SCOPE_KEY, _CACHE_SCOPE_LABEL + owner_id.encode("utf-8"), hashlib.sha256).hexdigest()
    token = _CACHE_SCOPE.set(scope)
    try:
        yield
    finally:
        _CACHE_SCOPE.reset(token)


def cache_scope_headers() -> dict[str, str]:
    """Return the header a chat call carries for its owner's cache scope.

    Only chat calls send it: the other endpoints keep no prompt cache.

    Returns:
        dict[str, str]: ``{"X-Cache-Scope": <64 hex characters>}`` inside
            :func:`cache_scope` with an owner, otherwise empty.
    """
    scope = _CACHE_SCOPE.get()
    return {CACHE_SCOPE_HEADER: scope} if scope else {}


def reply_usage(reply: Any) -> tuple[int | None, int | None]:
    """Read the token counts an OpenAI-compatible reply reports.

    Args:
        reply: A reply object from the ``openai`` SDK (chat completion or
            transcription).

    Returns:
        tuple[int | None, int | None]: ``(input_tokens, output_tokens)``;
            ``(None, None)`` when the reply carries no token usage.
    """
    usage = getattr(reply, "usage", None)
    if usage is None:
        return None, None
    prompt = getattr(usage, "prompt_tokens", None)
    if not isinstance(prompt, int):
        prompt = getattr(usage, "input_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    if not isinstance(completion, int):
        completion = getattr(usage, "output_tokens", None)
    if isinstance(prompt, int) and isinstance(completion, int):
        return prompt, completion
    return None, None
