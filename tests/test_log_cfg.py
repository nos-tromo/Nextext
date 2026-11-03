"""Tests for the log helpers and uvicorn filters in ``nextext.utils.log_cfg``.

Operational logs must not carry request or transcript content: an exception is
described by class (and HTTP status) and frames, never by its message, and
uvicorn's access and error records lose the query string and the exception
message respectively.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from typing import override

import httpx
import httpx2
import openai
import pytest
from uvicorn.config import LOGGING_CONFIG
from uvicorn.logging import AccessFormatter

from nextext.utils import log_cfg
from nextext.utils.log_cfg import error_label, exception_frames, setup_logging

# Invented stand-ins for content an upstream error could echo and for a media
# capability token.
_ECHOED = "invented transcript line 0427"
_TOKEN = "invented-media-token-7f3a"
_MEDIA_PATH = "/api/v1/jobs/job-0001/media"


def _raised(exc: Exception) -> Exception:
    """Raise ``exc`` and return it with its traceback attached.

    Args:
        exc (Exception): The exception to raise.

    Returns:
        Exception: The same exception, now carrying a traceback.
    """
    try:
        raise exc
    except Exception as caught:
        return caught


def _access_formatter() -> AccessFormatter:
    """Build uvicorn's access formatter with its default format, uncoloured.

    Returns:
        AccessFormatter: The formatter uvicorn applies to access records.
    """
    return AccessFormatter(fmt=LOGGING_CONFIG["formatters"]["access"]["fmt"], use_colors=False)


def _access_record(path: str) -> logging.LogRecord:
    """Build an access record shaped exactly like uvicorn's.

    Args:
        path (str): The path-with-query argument uvicorn would log.

    Returns:
        logging.LogRecord: A ``uvicorn.access`` record.
    """
    return logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg='%s - "%s %s HTTP/%s" %d',
        args=("10.0.0.2:51234", "GET", path, "1.1", 206),
        exc_info=None,
    )


class _ListHandler(logging.Handler):
    """Stdlib handler that keeps each formatted record."""

    def __init__(self, formatter: logging.Formatter) -> None:
        """Create the handler.

        Args:
            formatter (logging.Formatter): Formatter applied to each record.
        """
        super().__init__(level=logging.DEBUG)
        self.setFormatter(formatter)
        self.lines: list[str] = []

    @override
    def emit(self, record: logging.LogRecord) -> None:
        """Store the formatted record.

        Args:
            record (logging.LogRecord): The record to keep.
        """
        self.lines.append(self.format(record))


# ---------------------------------------------------------------------------
# error_label / exception_frames
# ---------------------------------------------------------------------------


def test_error_label_names_an_openai_error_by_class_and_status() -> None:
    """An SDK status error keeps its class and status, never its echoed body."""
    request = httpx2.Request("POST", "http://inference.invalid/v1/chat/completions")
    response = httpx2.Response(400, request=request, text=_ECHOED)
    exc = openai.BadRequestError(f"Error code: 400 - {_ECHOED}", response=response, body={"error": _ECHOED})

    assert error_label(exc) == "BadRequestError (HTTP 400)"


def test_error_label_reads_the_status_of_an_httpx_error() -> None:
    """``httpx.HTTPStatusError`` carries its status on the response."""
    request = httpx.Request("POST", "http://router.invalid/gliner")
    response = httpx.Response(422, request=request, text=_ECHOED)
    exc = httpx.HTTPStatusError(_ECHOED, request=request, response=response)

    assert error_label(exc) == "HTTPStatusError (HTTP 422)"


@pytest.mark.parametrize("exc", [httpx.ConnectError(_ECHOED), ValueError(_ECHOED), RuntimeError(_ECHOED)])
def test_error_label_is_the_bare_class_without_a_status(exc: Exception) -> None:
    """An error with no HTTP status is named by its class alone.

    Args:
        exc (Exception): An error whose message quotes invented content.
    """
    assert error_label(exc) == type(exc).__name__


def test_exception_frames_locate_the_failure_without_its_message() -> None:
    """The frames name where the error was raised; the message is left out."""

    def _fails_deep_inside() -> None:
        """Raise an error whose message quotes invented content.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError(_ECHOED)

    with pytest.raises(RuntimeError) as raised:
        _fails_deep_inside()
    frames = exception_frames(raised.value)

    assert frames.startswith("Traceback (most recent call last):\n")
    assert "_fails_deep_inside" in frames
    assert _ECHOED not in frames


def test_exception_frames_are_empty_for_an_exception_never_raised() -> None:
    """An exception that was never raised has no frames to render."""
    assert exception_frames(RuntimeError(_ECHOED)) == ""


# ---------------------------------------------------------------------------
# uvicorn filters
# ---------------------------------------------------------------------------


def test_access_filter_drops_the_query_string() -> None:
    """The media token never reaches the formatted access line."""
    record = _access_record(f"{_MEDIA_PATH}?token={_TOKEN}")

    assert log_cfg._StripQueryString().filter(record) is True
    line = _access_formatter().format(record)

    assert f'"GET {_MEDIA_PATH} HTTP/1.1" 206' in line
    assert _TOKEN not in line
    assert "?" not in line


def test_access_filter_leaves_other_records_alone() -> None:
    """A path without a query, or a record of another shape, passes unchanged."""
    plain = _access_record(_MEDIA_PATH)
    no_args = logging.LogRecord("uvicorn.access", logging.INFO, __file__, 1, "plain message", None, None)
    short_args = logging.LogRecord("uvicorn.access", logging.INFO, __file__, 1, "%s?", ("x?y",), None)

    for record in (plain, no_args, short_args):
        before = record.args
        assert log_cfg._StripQueryString().filter(record) is True
        assert record.args == before


def test_error_filter_keeps_class_and_frames_but_not_the_message() -> None:
    """A re-raised route error is logged without the content its message quotes."""
    exc = _raised(RuntimeError(_ECHOED))
    record = logging.LogRecord(
        "uvicorn.error",
        logging.ERROR,
        __file__,
        1,
        "Exception in ASGI application\n",
        (),
        (type(exc), exc, exc.__traceback__),
    )

    assert log_cfg._StripExceptionMessage().filter(record) is True
    line = logging.Formatter("%(message)s").format(record)

    assert line.startswith("Exception in ASGI application: RuntimeError\nTraceback (most recent call last):\n")
    assert "_raised" in line
    assert _ECHOED not in line


def test_error_filter_leaves_records_without_an_exception_alone() -> None:
    """Ordinary error-log records are not rewritten."""
    record = logging.LogRecord("uvicorn.error", logging.INFO, __file__, 1, "Started server process [%d]", (7,), None)

    assert log_cfg._StripExceptionMessage().filter(record) is True
    assert record.getMessage() == "Started server process [7]"


def test_setup_logging_scrubs_uvicorn_records_and_attaches_once() -> None:
    """Repeated setup attaches one filter per logger, and real emits are scrubbed."""
    setup_logging()
    setup_logging()
    access = logging.getLogger("uvicorn.access")
    error = logging.getLogger("uvicorn.error")

    assert sum(isinstance(f, log_cfg._StripQueryString) for f in access.filters) == 1
    assert sum(isinstance(f, log_cfg._StripExceptionMessage) for f in error.filters) == 1

    access_handler = _ListHandler(_access_formatter())
    error_handler = _ListHandler(logging.Formatter("%(message)s"))
    saved_levels = (access.level, error.level)
    access.addHandler(access_handler)
    error.addHandler(error_handler)
    access.setLevel(logging.INFO)
    error.setLevel(logging.INFO)
    try:
        access.info('%s - "%s %s HTTP/%s" %d', "10.0.0.2:51234", "GET", f"{_MEDIA_PATH}?token={_TOKEN}", "1.1", 206)
        error.error("Exception in ASGI application\n", exc_info=_raised(RuntimeError(_ECHOED)))
    finally:
        access.removeHandler(access_handler)
        error.removeHandler(error_handler)
        access.setLevel(saved_levels[0])
        error.setLevel(saved_levels[1])

    assert len(access_handler.lines) == 1
    assert f'"GET {_MEDIA_PATH} HTTP/1.1"' in access_handler.lines[0]
    assert _TOKEN not in access_handler.lines[0]
    assert len(error_handler.lines) == 1
    assert "RuntimeError" in error_handler.lines[0]
    assert _ECHOED not in error_handler.lines[0]


def test_setup_logging_moves_uvicorn_off_stdout() -> None:
    """Uvicorn's default config logs access to stdout, which belongs to the audit log; setup moves it."""
    assert LOGGING_CONFIG["handlers"]["access"]["stream"] == "ext://sys.stdout"
    access = logging.getLogger("uvicorn.access")
    handler = logging.StreamHandler(sys.stdout)
    access.addHandler(handler)
    try:
        setup_logging()
        assert handler.stream is sys.stderr
    finally:
        access.removeHandler(handler)


def test_uvicorn_access_log_never_reaches_stdout_in_a_real_process() -> None:
    """Under uvicorn's own default logging config, an access line lands on stderr and stdout stays empty."""
    code = (
        "import logging, logging.config\n"
        "from uvicorn.config import LOGGING_CONFIG\n"
        "logging.config.dictConfig(LOGGING_CONFIG)\n"
        "from nextext.utils.log_cfg import setup_logging\n"
        "setup_logging()\n"
        "logging.getLogger('uvicorn.access').info("
        "'%s - \"%s %s HTTP/%s\" %d', '10.0.0.2:51234', 'GET', '/api/v1/health', '1.1', 200)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=True)
    assert result.stdout == ""
    assert '"GET /api/v1/health HTTP/1.1" 200' in result.stderr
