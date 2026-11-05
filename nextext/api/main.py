"""FastAPI application factory and uvicorn entry point for the Nextext backend."""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any

from fastapi import FastAPI
from loguru import logger
from prometheus_fastapi_instrumentator import Instrumentator

from nextext.api import cli_files
from nextext.api.errors import install_error_handlers
from nextext.api.identity import check_principal_config
from nextext.api.jobs import JobManager, JobState, PushEvent
from nextext.api.routes import router as api_router
from nextext.audit import build_emitter, emit_config_event, service_version
from nextext.utils.audit_event import AuditEmitter
from nextext.utils.audit_event import AuditWriteError
from nextext.utils.env_cfg import CliRetentionConfig, load_audit_env, load_cli_retention_env, load_principal_env
from nextext.utils.log_cfg import setup_logging

_APP_VERSION = service_version()

#: Builds the process's audit emitter from the ``AUDIT_OUTPUT`` mode.
EmitterFactory = Callable[[str | None], AuditEmitter]
#: The blocking pipeline a job runs (see :class:`nextext.api.jobs.JobManager`).
PipelineRunner = Callable[[JobState, PushEvent], dict[str, Any]]


#: How often the backend sweeps what CLI runs left, after the sweep at startup.
CLI_SWEEP_EVERY_S = 86_400.0


def _cli_sweeper(config: CliRetentionConfig) -> cli_files.CliFileSweeper:
    """Build the sweeper of CLI runs' files (paths read now, so tests can point them elsewhere).

    Args:
        config: The retention settings.

    Returns:
        cli_files.CliFileSweeper: The sweeper.
    """
    return cli_files.CliFileSweeper(
        cli_files.CLI_ROOT, cli_files.LEGACY_OUTPUT, days=config.days, notice_days=config.notice_days
    )


def _sweep_cli_files(sweeper: cli_files.CliFileSweeper, emitter: AuditEmitter) -> None:
    """Sweep once; a damaged state or an unreadable directory is logged and retried next time.

    Args:
        sweeper: The sweeper.
        emitter: The backend's audit emitter.

    Raises:
        AuditWriteError: If a deletion's record cannot be written; nothing was deleted.
    """
    try:
        result = sweeper.sweep(time.time(), emitter)
    except (OSError, ValueError) as exc:
        logger.error("CLI-file retention could not run ({}); nothing was deleted.", type(exc).__name__)
        return
    if result.waiting_until is not None:
        logger.info(
            "CLI-file retention takes effect on {}; until then nothing is deleted.",
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(result.waiting_until)),
        )


async def _cli_sweep_loop(sweeper: cli_files.CliFileSweeper, emitter: AuditEmitter) -> None:
    """Sweep CLI runs' files daily until cancelled.

    Args:
        sweeper: The sweeper.
        emitter: The backend's audit emitter.
    """
    while True:
        await asyncio.sleep(CLI_SWEEP_EVERY_S)
        try:
            await asyncio.to_thread(_sweep_cli_files, sweeper, emitter)
        except AuditWriteError:
            logger.error("A CLI-file deletion could not be recorded; nothing was deleted.")


def _lifespan(
    emitter_factory: EmitterFactory,
    pipeline_runner: PipelineRunner | None,
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    """Build the app's lifespan around an emitter factory and a pipeline runner.

    Args:
        emitter_factory: Creates the process's audit emitter.
        pipeline_runner: The job pipeline; ``None`` runs the real one.

    Returns:
        Callable[[FastAPI], AbstractAsyncContextManager[None]]: The lifespan.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Configure logging, open the audit chain, and run the job manager.

        The audit chain opens before the job manager starts — whose first act
        is to remove, and record removing, the uploads a previous process
        left behind — and closes after it stops.

        Args:
            app: The FastAPI application instance.

        Yields:
            None: Control is yielded after startup configuration completes.

        Raises:
            PrincipalConfigError: When the identity configuration is malformed,
                so the backend refuses to start.
            ValueError: When ``AUDIT_REQUIRE_PURPOSE`` or ``AUDIT_OUTPUT``
                names something unknown.
            OSError: When the audit stream cannot be opened.
        """
        setup_logging()
        check_principal_config()
        audit_cfg = load_audit_env()
        principal_cfg = load_principal_env()
        cli_retention = load_cli_retention_env()
        emitter = emitter_factory(audit_cfg.output)
        emitter.start()
        try:
            # The federation-wide keys, plus how many services may call as
            # themselves (a count — names and tokens stay out).
            emit_config_event(
                emitter,
                {
                    "edge_secret_enforced": principal_cfg.edge_secret is not None,
                    "default_identity_enabled": principal_cfg.default_identity is not None
                    and principal_cfg.edge_secret is None,
                    "require_purpose": sorted(audit_cfg.require_purpose),
                    "service_tokens": len(principal_cfg.service_tokens),
                    "cli_retention_days": cli_retention.days,
                    "retention_notice_days": cli_retention.notice_days,
                },
            )
            job_manager = JobManager(pipeline_runner=pipeline_runner, emitter=emitter)
            await job_manager.start()
            cli_sweep: asyncio.Task[None] | None = None
            if job_manager.holds_root_lock and cli_retention.days is not None:
                sweeper = _cli_sweeper(cli_retention)
                await asyncio.to_thread(_sweep_cli_files, sweeper, emitter)
                cli_sweep = asyncio.create_task(_cli_sweep_loop(sweeper, emitter))
            app.state.audit_emitter = emitter
            app.state.audit_required = audit_cfg.require_purpose
            app.state.job_manager = job_manager
            logger.info("Nextext API started")
            try:
                yield
            finally:
                logger.info("Nextext API shutting down.")
                if cli_sweep is not None:
                    cli_sweep.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await cli_sweep
                await job_manager.stop()
        finally:
            emitter.close()

    return lifespan


def create_app(
    *,
    audit_emitter_factory: EmitterFactory = build_emitter,
    pipeline_runner: PipelineRunner | None = None,
) -> FastAPI:
    """Build the FastAPI application, mount routers, and return it.

    Args:
        audit_emitter_factory: Creates the process's audit emitter from the
            ``AUDIT_OUTPUT`` mode; tests pass one writing to a buffer.
        pipeline_runner: The job pipeline; ``None`` runs the real one.

    Returns:
        FastAPI: A configured application ready to be served by uvicorn.
    """
    application = FastAPI(
        title="Nextext API",
        description=(
            "Backend for the Nextext audio transcription, translation, and "
            "analysis toolkit. The React frontend in this repository is "
            "one consumer; any HTTP client can use the documented endpoints."
        ),
        version=_APP_VERSION,
        lifespan=_lifespan(audit_emitter_factory, pipeline_runner),
    )
    install_error_handlers(application)
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
