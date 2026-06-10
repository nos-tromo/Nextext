"""Integration tests for ownership-scoped, in-memory job routes."""

from __future__ import annotations

import io
import json
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from nextext.api.jobs import JobManager
from nextext.api.main import create_app

from .conftest import (
    ALICE_OWNER_ID,
    BOB_OWNER_ID,
    _client_with_owner,
    stub_pipeline_runner,
)


@pytest.fixture
def dual_clients() -> Iterator[tuple[TestClient, TestClient]]:
    """Spin up one stubbed backend shared by two owner-scoped clients.

    The two clients model two separate browsers (distinct identities)
    talking to the same in-memory ``JobManager``.

    Yields:
        tuple[TestClient, TestClient]: ``(alice, bob)`` clients.
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
        with (
            _client_with_owner(app, ALICE_OWNER_ID) as alice,
            _client_with_owner(app, BOB_OWNER_ID) as bob,
        ):
            yield alice, bob
    finally:
        app.router.lifespan_context = original_lifespan


def _submit(client: TestClient, *, name: str = "clip.wav") -> str:
    """Submit a stub job and return its id.

    Args:
        client: TestClient bound to the dual-client fixture.
        name: Filename used in the upload form.

    Returns:
        str: The new job id.
    """
    options = {
        "task": "transcribe",
        "trg_lang": "de",
        "speakers": 1,
        "words": False,
        "summarization": False,
        "hate_speech": False,
    }
    files = {"file": (name, io.BytesIO(b"x"), "audio/wav")}
    response = client.post("/api/v1/jobs", files=files, data={"options": json.dumps(options)})
    assert response.status_code == 201, response.text
    return cast(str, response.json()["job_id"])


def _wait_for_completed(client: TestClient, job_id: str) -> dict[str, Any]:
    """Poll ``GET /jobs/{id}`` until completion.

    Args:
        client: TestClient.
        job_id: Job identifier.

    Returns:
        dict[str, Any]: Final snapshot body.
    """
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/jobs/{job_id}")
        if response.status_code == 200 and response.json()["status"] == "completed":
            return cast(dict[str, Any], response.json())
        time.sleep(0.05)
    raise AssertionError(f"Job {job_id} never completed.")


def test_list_jobs_is_owner_scoped(dual_clients: tuple[TestClient, TestClient]) -> None:
    """Each owner sees only their own jobs in the listing."""
    alice, bob = dual_clients
    alice_id = _submit(alice, name="alice.wav")
    bob_id = _submit(bob, name="bob.wav")
    _wait_for_completed(alice, alice_id)
    _wait_for_completed(bob, bob_id)

    alice_list = alice.get("/api/v1/jobs").json()["jobs"]
    bob_list = bob.get("/api/v1/jobs").json()["jobs"]

    assert [j["job_id"] for j in alice_list] == [alice_id]
    assert [j["job_id"] for j in bob_list] == [bob_id]


def test_all_owner_jobs_appear_in_listing(dual_clients: tuple[TestClient, TestClient]) -> None:
    """Every job the caller owns is listed — the basis for reload re-discovery."""
    alice, _ = dual_clients
    first = _submit(alice, name="one.wav")
    second = _submit(alice, name="two.wav")
    _wait_for_completed(alice, first)
    _wait_for_completed(alice, second)
    ids = {j["job_id"] for j in alice.get("/api/v1/jobs").json()["jobs"]}
    assert {first, second} <= ids


def test_cross_owner_get_and_delete_return_404(dual_clients: tuple[TestClient, TestClient]) -> None:
    """One owner must never see or delete another owner's job by id."""
    alice, bob = dual_clients
    alice_id = _submit(alice)
    _wait_for_completed(alice, alice_id)
    assert bob.get(f"/api/v1/jobs/{alice_id}").status_code == 404
    assert bob.delete(f"/api/v1/jobs/{alice_id}").status_code == 404
    # Alice still owns and sees her own job.
    assert alice.get(f"/api/v1/jobs/{alice_id}").status_code == 200


def test_artifact_download_round_trips(dual_clients: tuple[TestClient, TestClient]) -> None:
    """Downloading a CSV artifact after completion succeeds (from memory)."""
    alice, _ = dual_clients
    job_id = _submit(alice)
    _wait_for_completed(alice, job_id)
    response = alice.get(f"/api/v1/jobs/{job_id}/artifacts/transcript.csv")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert b"Hello world." in response.content


def test_delete_removes_job(dual_clients: tuple[TestClient, TestClient]) -> None:
    """DELETE drops the job; subsequent reads 404 and it leaves the listing."""
    alice, _ = dual_clients
    job_id = _submit(alice)
    _wait_for_completed(alice, job_id)
    assert alice.delete(f"/api/v1/jobs/{job_id}").status_code == 204
    assert alice.get(f"/api/v1/jobs/{job_id}").status_code == 404
    assert job_id not in {j["job_id"] for j in alice.get("/api/v1/jobs").json()["jobs"]}
