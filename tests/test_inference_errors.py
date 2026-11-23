"""Tests for the shared classifier of inference outages."""

from __future__ import annotations

import httpx2
import openai
import pytest

from nextext.core.inference_errors import InferenceUnreachable, find_inference_outage, is_transient_inference_error

_REQUEST = httpx2.Request("POST", "http://router.invalid/v1/chat/completions")


def _status_error(status_code: int) -> openai.APIStatusError:
    """Build the SDK error for an HTTP error status.

    Args:
        status_code (int): The response status.

    Returns:
        openai.APIStatusError: The error.
    """
    response = httpx2.Response(status_code, request=_REQUEST)
    return openai.APIStatusError(f"Error code: {status_code}", response=response, body=None)


@pytest.mark.parametrize(
    ("status_code", "transient"), [(400, False), (401, False), (404, False), (429, True), (500, True), (503, True)]
)
def test_only_retryable_statuses_are_transient(status_code: int, transient: bool) -> None:
    """HTTP 429 and 5xx are transient; the other 4xx are the request's own fault."""
    assert is_transient_inference_error(_status_error(status_code)) is transient


def test_connection_failures_and_timeouts_are_transient() -> None:
    """The SDK's connection error and its timeout subclass are transient."""
    assert is_transient_inference_error(openai.APIConnectionError(request=_REQUEST))
    assert is_transient_inference_error(openai.APITimeoutError(request=_REQUEST))
    assert not is_transient_inference_error(RuntimeError("bug"))


def test_the_outage_is_returned_whether_raised_directly_or_further_down_the_chain() -> None:
    """The nearest outage of the chain comes back, so a caller can log its class and status."""
    outage = _status_error(503)
    assert find_inference_outage(outage) is outage

    try:
        try:
            raise outage
        except openai.APIStatusError as caught:
            raise RuntimeError("wrapped") from caught
    except RuntimeError as wrapped:
        assert find_inference_outage(wrapped) is outage

    unreachable = InferenceUnreachable("provider unreachable")
    assert find_inference_outage(unreachable) is unreachable
    assert isinstance(unreachable, ConnectionError)


@pytest.mark.parametrize(
    "failure",
    [
        ConnectionError("plain"),
        BrokenPipeError(),
        ConnectionResetError(),
        ConnectionRefusedError(),
        ConnectionAbortedError(),
    ],
    ids=lambda failure: type(failure).__name__,
)
def test_the_builtin_connection_errors_are_no_outage(failure: ConnectionError) -> None:
    """Only the dedicated class counts: a closed stream or a reset socket may be anything's."""
    assert find_inference_outage(failure) is None

    try:
        try:
            raise failure
        except ConnectionError as caught:
            raise RuntimeError("wrapped") from caught
    except RuntimeError as wrapped:
        assert find_inference_outage(wrapped) is None


def test_no_outage_in_the_chain_returns_none() -> None:
    """A rejected request, or a bug, is not an outage, however deep."""
    try:
        try:
            raise _status_error(400)
        except openai.APIStatusError as caught:
            raise RuntimeError("wrapped") from caught
    except RuntimeError as wrapped:
        assert find_inference_outage(wrapped) is None


def test_a_context_the_raiser_suppressed_is_not_followed_but_a_cause_always_is() -> None:
    """``raise ... from None`` says the earlier error is unrelated."""
    try:
        try:
            raise _status_error(503)
        except openai.APIStatusError:
            raise RuntimeError("unrelated") from None
    except RuntimeError as unrelated:
        assert find_inference_outage(unrelated) is None


def test_a_chain_that_loops_back_on_itself_terminates() -> None:
    """Two exceptions naming each other as context do not hang the walk."""
    first = RuntimeError("first")
    second = RuntimeError("second")
    first.__context__ = second
    second.__context__ = first

    assert find_inference_outage(first) is None
