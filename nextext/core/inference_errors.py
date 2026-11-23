"""Telling an inference outage from every other failure.

The pipeline degrades gracefully on a transient failure of a model call; the
job worker names one a typed failure code. Both ask the same question, so the
answer lives here, once.
"""

import openai


class InferenceUnreachable(ConnectionError):
    """The configured inference provider did not answer its health check.

    Raised by the job worker at the start of a stage that needs the chat
    provider, and the cause of the ``RuntimeError`` that
    :meth:`nextext.core.openai_cfg.InferencePipeline._chat` raises when the
    provider drops between two requests. It is a :class:`ConnectionError` for
    callers that catch that, but
    only this class counts as an outage in :func:`find_inference_outage`: the
    builtin family also holds a ``BrokenPipeError`` from a closed output
    stream, which is the audit chain's failure and no inference outage.
    """


def is_transient_inference_error(exc: BaseException) -> bool:
    """Report whether an exception is a transient inference-service failure.

    Transient means the request itself was well-formed but the service could
    not serve it right now: client-side connection failures (including
    timeouts) and retryable server statuses (HTTP 429 and 5xx, e.g. a router
    that cannot reach its upstream model). Configuration-shaped errors
    (4xx such as 401/404) are not transient and must surface loudly.

    Args:
        exc (BaseException): The exception raised by an inference call.

    Returns:
        bool: ``True`` when the failure is transient and safe to degrade on.
    """
    if isinstance(exc, openai.APIConnectionError):
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code == 429 or exc.status_code >= 500
    return False


def find_inference_outage(exc: BaseException) -> BaseException | None:
    """Find the inference outage a failure is, or was raised from.

    A stage often wraps the SDK's error in its own (``raise RuntimeError(...)
    from exc``), so the exception, its ``__cause__`` and its ``__context__``
    (unless ``raise ... from None`` suppressed it) are all asked. An outage is
    a transient inference failure (:func:`is_transient_inference_error`) or
    :class:`InferenceUnreachable`, which is what the job worker raises when the
    configured provider is unreachable at the start of a stage, and what
    ``_chat`` chains to its ``RuntimeError`` when it drops between two
    requests. Any other :class:`ConnectionError` is not one.

    Args:
        exc (BaseException): The failure to classify.

    Returns:
        BaseException | None: The first exception of the chain, nearest the
            failure, that is an outage; ``None`` when the failure is not one.
            Callers log it with ``error_label``: its class and status say
            more than the wrapper's.
    """
    seen: set[int] = set()
    pending: list[BaseException] = [exc]
    while pending:
        current = pending.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, InferenceUnreachable) or is_transient_inference_error(current):
            return current
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None and not current.__suppress_context__:
            pending.append(current.__context__)
    return None
