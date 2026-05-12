"""Shared fixtures for FastAPI backend tests."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import pytest
from fastapi.testclient import TestClient

from nextext.api.jobs import (
    JobManager,
    JobState,
    PIPELINE_STAGE_LABELS,
    PushEvent,
)
from nextext.api.main import create_app


@pytest.fixture
def api_client() -> Iterator[TestClient]:
    """Provide a TestClient backed by a fresh FastAPI app instance.

    Yields:
        TestClient: A client that exercises the full app including the
            lifespan startup hook (logging + offline-mode env).
    """
    app = create_app()
    with TestClient(app) as client:
        yield client


def _fake_transcript_df() -> pd.DataFrame:
    """Build a tiny deterministic transcript DataFrame.

    Returns:
        pd.DataFrame: A transcript with two segments.
    """
    return pd.DataFrame(
        {
            "start": ["00:00:00", "00:00:02"],
            "end": ["00:00:02", "00:00:04"],
            "speaker": ["S1", "S2"],
            "text": ["Hello world.", "Second segment."],
        }
    )


def stub_pipeline_runner(state: JobState, push: PushEvent) -> dict[str, Any]:
    """Deterministic stand-in for the real pipeline, used across API tests.

    Args:
        state: The job being processed.
        push: Event sink for SSE delivery.

    Returns:
        dict[str, Any]: A result dict shaped like the real pipeline's output.
    """
    df = _fake_transcript_df()
    total = len(PIPELINE_STAGE_LABELS)
    for index, label in enumerate(PIPELINE_STAGE_LABELS):
        push(
            "stage_started",
            {
                "stage": label,
                "stage_index": index,
                "progress": index / total,
                "timestamp": "stub",
            },
        )
        push(
            "stage_completed",
            {
                "stage": label,
                "stage_index": index,
                "progress": (index + 1) / total,
                "timestamp": "stub",
                "result_delta": {"index": index},
            },
        )
    return {
        "transcript": df,
        "summary": "A short summary." if state.options.summarization else None,
        "word_counts": None,
        "named_entities": None,
        "wordcloud": None,
        "hate_speech_findings": None,
        "resolved_src_lang": "en",
        "transcript_language": "en",
        "task": state.options.task,
    }


@pytest.fixture
def stub_app_client() -> Iterator[tuple[TestClient, JobManager]]:
    """Spin up the FastAPI app with the stubbed pipeline runner.

    Yields:
        tuple[TestClient, JobManager]: The HTTP client and the manager it uses.
    """
    app = create_app()
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def _patched_lifespan(_app):  # type: ignore[no-untyped-def]
        manager = JobManager(
            pipeline_runner=stub_pipeline_runner,
            ttl_seconds=3600,
        )
        await manager.start()
        _app.state.job_manager = manager
        try:
            yield
        finally:
            await manager.stop()

    app.router.lifespan_context = _patched_lifespan
    try:
        with TestClient(app) as client:
            yield client, client.app.state.job_manager  # type: ignore[union-attr]
    finally:
        app.router.lifespan_context = original_lifespan
