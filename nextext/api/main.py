"""FastAPI application factory and uvicorn entry point for the Nextext backend."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version

from fastapi import FastAPI
from loguru import logger
from prometheus_fastapi_instrumentator import Instrumentator

from nextext.api.jobs import JobManager
from nextext.api.routes import router as api_router
from nextext.utils.log_cfg import setup_logging

try:
    _APP_VERSION = version("nextext")
except PackageNotFoundError:  # running from source without an installed dist
    _APP_VERSION = "0+unknown"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Configure logging and the job manager for the app lifetime.

    Args:
        app: The FastAPI application instance.

    Yields:
        None: Control is yielded after startup configuration completes.
    """
    setup_logging()
    job_manager = JobManager()
    await job_manager.start()
    app.state.job_manager = job_manager
    logger.info("Nextext API started")
    try:
        yield
    finally:
        logger.info("Nextext API shutting down.")
        await job_manager.stop()


def create_app() -> FastAPI:
    """Build the FastAPI application, mount routers, and return it.

    Returns:
        FastAPI: A configured application ready to be served by uvicorn.
    """
    application = FastAPI(
        title="Nextext API",
        description=(
            "Backend for the Nextext audio transcription, translation, and "
            "analysis toolkit. The Streamlit frontend in this repository is "
            "one consumer; any HTTP client can use the documented endpoints."
        ),
        version=_APP_VERSION,
        lifespan=lifespan,
    )
    application.include_router(api_router)
    # Aggregate request/latency counters only — no transcript or user data is
    # ever recorded in a metric label or value. Unauthenticated by design:
    # the obs-plane scraper (like every inference caller) reaches the backend
    # over inference-net, which has no auth boundary today.
    Instrumentator().instrument(application).expose(application, endpoint="/metrics", include_in_schema=False)
    return application


app = create_app()


def run() -> None:
    """Run the API with uvicorn — entry point for ``nextext-api``."""
    import uvicorn

    host = os.getenv("NEXTEXT_API_HOST", "0.0.0.0")
    port = int(os.getenv("NEXTEXT_API_PORT", "8000"))
    uvicorn.run(
        "nextext.api.main:app",
        host=host,
        port=port,
        workers=1,
        log_config=None,
    )


if __name__ == "__main__":  # pragma: no cover
    run()
