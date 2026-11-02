"""Global error handlers: generic client-visible bodies, diagnosable logs.

An unhandled exception is logged by class and traceback frames, never by
message: the message can quote the content being processed.
"""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger

from nextext.utils.log_cfg import error_label, exception_frames


def install_error_handlers(app: FastAPI) -> None:
    """Register handlers that keep exception detail out of response bodies.

    Args:
        app: The FastAPI application to attach the handlers to.
    """

    @app.exception_handler(RequestValidationError)
    async def _handle_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        logger.warning(f"Validation error on {request.method} {request.url.path}: {exc.errors()}")
        return JSONResponse(status_code=422, content={"detail": "Invalid request."})

    @app.exception_handler(Exception)
    async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.error(
            "Unhandled error on {} {}: {}\n{}",
            request.method,
            request.url.path,
            error_label(exc),
            exception_frames(exc),
        )
        return JSONResponse(status_code=500, content={"detail": "Internal server error."})
