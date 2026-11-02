"""Logging setup for the Nextext backend and CLI.

Installs the loguru stderr sink and keeps request and media content out of
the operational logs: an upstream or model error is logged by class and HTTP
status (``error_label``), an unexpected failure by class and traceback frames
(``exception_frames``) — never by message, which can quote the data being
processed. The uvicorn access log drops the query string, which can carry the
per-job media token.

The container logging driver owns log retention and rotation
(see ``docker/compose.yaml``).
"""

from __future__ import annotations

import logging
import os
import sys
import traceback
from typing import override

from loguru import logger


def error_label(exc: BaseException) -> str:
    """Name an exception for the logs without quoting its message.

    An upstream or model error message can quote the request it rejected — a
    transcript chunk, a prompt — so only the exception class and, for an HTTP
    error, its status code are kept.

    Args:
        exc (BaseException): The exception to describe.

    Returns:
        str: ``"<Class>"``, or ``"<Class> (HTTP <status>)"`` when the exception
            carries a status code (``openai.APIStatusError``, or an HTTP error
            whose ``response`` has one, e.g. ``httpx.HTTPStatusError``).
    """
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        status = getattr(getattr(exc, "response", None), "status_code", None)
    name = type(exc).__name__
    return f"{name} (HTTP {status})" if isinstance(status, int) else name


def exception_frames(exc: BaseException) -> str:
    """Render an exception's traceback frames without its message.

    The frames say where a failure happened; the message is left out because
    it can quote the data being processed.

    Args:
        exc (BaseException): The exception whose traceback to render.

    Returns:
        str: ``Traceback (most recent call last):`` followed by the frames, or
            an empty string when the exception carries no traceback.
    """
    frames = "".join(traceback.format_tb(exc.__traceback__)).rstrip()
    return f"Traceback (most recent call last):\n{frames}" if frames else ""


class _StripQueryString(logging.Filter):
    """Drop the query string from uvicorn access-log records.

    uvicorn logs each request with ``record.args == (client_addr, method,
    path_with_query, http_version, status)``. The query string can carry the
    per-job media token, so only the path is kept.
    """

    @override
    def filter(self, record: logging.LogRecord) -> bool:
        """Rewrite the record's path argument in place.

        Args:
            record (logging.LogRecord): An access-log record.

        Returns:
            bool: Always ``True``; the record is kept, only rewritten.
        """
        args = record.args
        if isinstance(args, tuple) and len(args) >= 3:
            path = args[2]
            if isinstance(path, str) and "?" in path:
                record.args = (*args[:2], path.split("?", 1)[0], *args[3:])
        return True


class _StripExceptionMessage(logging.Filter):
    """Keep exception messages out of uvicorn's error log.

    Starlette re-raises an unhandled route exception after the app's own
    handler has answered, and uvicorn then logs it with a full traceback. The
    rewritten record keeps the exception class and frames, not the message.
    """

    @override
    def filter(self, record: logging.LogRecord) -> bool:
        """Fold a record's exception into its message as class and frames.

        Args:
            record (logging.LogRecord): A ``uvicorn.error`` record.

        Returns:
            bool: Always ``True``; the record is kept, only rewritten.
        """
        exc = record.exc_info[1] if record.exc_info else None
        if exc is not None:
            summary = f"{record.getMessage().rstrip()}: {error_label(exc)}"
            frames = exception_frames(exc)
            record.msg = f"{summary}\n{frames}" if frames else summary
            record.args = ()
            record.exc_info = None
            record.exc_text = None
        return True


def _add_filter_once(logger_name: str, log_filter: logging.Filter) -> None:
    """Attach a filter to a stdlib logger unless one of its type is already there.

    Args:
        logger_name (str): Name of the stdlib logger.
        log_filter (logging.Filter): The filter to attach.
    """
    target = logging.getLogger(logger_name)
    if not any(type(existing) is type(log_filter) for existing in target.filters):
        target.addFilter(log_filter)


def setup_logging(
    backtrace: bool = False,
    diagnose: bool = False,
) -> None:
    """Set up logging for the application.

    Installs a single loguru stderr sink. ``LOG_LEVEL`` selects the minimum
    level (default ``INFO``). Also attaches, once per process, the filters
    that keep query strings out of uvicorn's access log and exception messages
    out of its error log (inert in the CLI, which runs no uvicorn).

    Args:
        backtrace (bool, optional): Whether to include backtrace information. Defaults to False.
        diagnose (bool, optional): Whether to include diagnostic information. Defaults to False.
    """
    level = os.getenv("LOG_LEVEL", "INFO")

    logger.remove()

    logger.add(
        sink=sys.stderr,
        level=level,
        backtrace=backtrace,
        diagnose=diagnose,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name} | {message}",
    )

    _add_filter_once("uvicorn.access", _StripQueryString())
    _add_filter_once("uvicorn.error", _StripExceptionMessage())
