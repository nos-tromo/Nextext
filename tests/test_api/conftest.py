"""Shared fixtures for FastAPI backend tests."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nextext.api.jobs import (
    PIPELINE_STAGE_LABELS,
    JobManager,
    JobState,
    PushEvent,
)
from nextext.api.main import create_app

# Default trusted identity header (matches the backend's NEXTEXT_AUTH_HEADER
# default). Tests send the owner id under this header on every request.
OWNER_HEADER = "X-Auth-User"

# Two stable, distinct owner ids for fixtures that need to model two
# different browsers without minting random values inside tests.
ALICE_OWNER_ID = "a" * 32
BOB_OWNER_ID = "b" * 32


def _client_with_owner(app: FastAPI, owner_id: str) -> TestClient:
    """Build a TestClient that sends ``owner_id`` on every request.

    Args:
        app: The FastAPI application to wrap.
        owner_id: Identity value sent as the trusted identity header on
            outgoing requests.

    Returns:
        TestClient: A client preconfigured with the owner header.
    """
    client = TestClient(app)
    client.headers[OWNER_HEADER] = owner_id
    return client


@pytest.fixture
def api_client() -> Iterator[TestClient]:
    """Provide a TestClient backed by a fresh FastAPI app instance.

    Every request from this client carries a fixed ``X-Owner-Id`` header
    so the ownership-aware routes resolve successfully without each test
    having to wire it manually.

    Yields:
        TestClient: A client that exercises the full app including the
            lifespan startup hook (logging + offline-mode env).
    """
    app = create_app()
    with _client_with_owner(app, ALICE_OWNER_ID) as client:
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
    async def _patched_lifespan(_app: Any) -> AsyncIterator[None]:
        manager = JobManager(pipeline_runner=stub_pipeline_runner)
        await manager.start()
        _app.state.job_manager = manager
        try:
            yield
        finally:
            await manager.stop()

    app.router.lifespan_context = _patched_lifespan
    try:
        with _client_with_owner(app, ALICE_OWNER_ID) as client:
            yield client, client.app.state.job_manager  # type: ignore[attr-defined]
    finally:
        app.router.lifespan_context = original_lifespan
