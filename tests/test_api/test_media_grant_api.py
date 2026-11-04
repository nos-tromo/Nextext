"""Tests for media playback through a grant (``POST /jobs/{id}/media-grant``).

A media element can send neither the identity header nor ``X-Audit-Purpose``,
so the SPA asks for a grant first and plays the URL it returns. These tests
pin the contract: the grant is owner-scoped and needs the purpose a query
needs, its issuance is not recorded, every stream played with it is recorded
with its purpose and its id (never the token), an unknown or expired grant is
refused unrecorded, and behind the gateway a grant plays only for the
identity it was issued to.
"""

from __future__ import annotations

import io
import json
import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nextext.api import main as main_module
from nextext.api.jobs import JobManager
from nextext.api.main import create_app
from nextext.api.media_grants import MediaGrants
from nextext.api.schemas import JobStatus
from nextext.utils.audit_runtime import encode_purpose
from tests.audit_helpers import AuditCapture, assert_no_content

from .conftest import ALICE_OWNER_ID, BOB_OWNER_ID, OWNER_HEADER, stub_pipeline_runner

MEDIA_BYTES = b"0123456789abcdef"
PURPOSE = encode_purpose("case-0001", "Review of a reported clip")
GRANT_REQUEST_ID = "7c9e6679-7425-40de-944b-e07fc1f90ae7"
RANGE_REQUEST_ID = "1b4e28ba-2fa1-41d2-883f-0016d3cca427"
EDGE_SECRET = "edge-secret-for-tests-only-0123456789abcdef"


@pytest.fixture
def build(monkeypatch: pytest.MonkeyPatch, audit_capture: AuditCapture) -> Any:
    """Return a builder of apps whose audit events go to ``audit_capture``.

    Args:
        monkeypatch: Pytest patcher.
        audit_capture: The capture.

    Returns:
        Any: ``build() -> FastAPI``, called after the test set its environment.
    """
    monkeypatch.setattr(main_module, "setup_logging", lambda: None)
    return lambda: create_app(audit_emitter_factory=audit_capture.factory, pipeline_runner=stub_pipeline_runner)


@pytest.fixture
def client(build: Any) -> Iterator[TestClient]:
    """Yield Alice's client on an app with the default settings.

    Args:
        build: The app builder.

    Yields:
        TestClient: The client.
    """
    with TestClient(build()) as test_client:
        test_client.headers[OWNER_HEADER] = ALICE_OWNER_ID
        yield test_client


def _manager(client: TestClient) -> JobManager:
    """Return the app's job manager.

    Args:
        client: A client of the app.

    Returns:
        JobManager: The manager.
    """
    app: FastAPI = client.app  # type: ignore[assignment]
    return app.state.job_manager


def _finished_job(client: TestClient, file_name: str = "clip.wav") -> str:
    """Create a job and wait until it has completed, without audited polling.

    Args:
        client: The owner's client.
        file_name: The upload name.

    Returns:
        str: The job id.
    """
    response = client.post(
        "/api/v1/jobs",
        files={"file": (file_name, io.BytesIO(MEDIA_BYTES), "audio/wav")},
        data={"options": json.dumps({"task": "transcribe"})},
    )
    assert response.status_code == 201, response.text
    job_id = str(response.json()["job_id"])
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        state = _manager(client)._jobs.get(job_id)
        if state is not None and state.status is JobStatus.COMPLETED:
            return job_id
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} did not complete")


def _grant(client: TestClient, job_id: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
    """Ask for a playback grant and return the response body.

    Args:
        client: The client.
        job_id: The job.
        headers: Extra request headers.

    Returns:
        dict[str, Any]: The grant response.
    """
    response = client.post(f"/api/v1/jobs/{job_id}/media-grant", headers=headers or {})
    assert response.status_code == 201, response.text
    return dict(response.json())


def _grant_token(media_url: str) -> str:
    """Return the grant token carried by a media URL.

    Args:
        media_url: The URL from a grant response.

    Returns:
        str: The token.
    """
    return parse_qs(urlparse(media_url).query)["grant"][0]


def _streams(audit_capture: AuditCapture) -> list[dict[str, Any]]:
    """Return the media.stream events written so far.

    Args:
        audit_capture: The capture.

    Returns:
        list[dict[str, Any]]: The events.
    """
    return audit_capture.of("query", "media.stream")


# --- issuance -----------------------------------------------------------------------


def test_a_grant_is_a_media_url_for_the_job(client: TestClient) -> None:
    """The grant comes back as a media URL for this job, uncached, expiring after the configured lifetime."""
    job_id = _finished_job(client)
    before = datetime.now(tz=UTC)
    response = client.post(f"/api/v1/jobs/{job_id}/media-grant")

    assert response.status_code == 201
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    parsed = urlparse(body["media_url"])
    assert parsed.path == f"/api/v1/jobs/{job_id}/media"
    assert len(_grant_token(body["media_url"])) >= 40
    expires_at = datetime.fromisoformat(body["expires_at"])
    assert before + timedelta(seconds=899) <= expires_at <= datetime.now(tz=UTC) + timedelta(seconds=901)


def test_the_grant_lifetime_is_configurable(monkeypatch: pytest.MonkeyPatch, build: Any) -> None:
    """NEXTEXT_MEDIA_GRANT_TTL_S sets how long a grant plays."""
    monkeypatch.setenv("NEXTEXT_MEDIA_GRANT_TTL_S", "60")
    with TestClient(build()) as client:
        client.headers[OWNER_HEADER] = ALICE_OWNER_ID
        expires_at = datetime.fromisoformat(_grant(client, _finished_job(client))["expires_at"])
    assert expires_at <= datetime.now(tz=UTC) + timedelta(seconds=61)


def test_issuing_a_grant_is_not_recorded(client: TestClient, audit_capture: AuditCapture) -> None:
    """The grant hands out no personal data: no event until it is used."""
    job_id = _finished_job(client)
    before = len(audit_capture.of())
    _grant(client, job_id, {"X-Audit-Purpose": PURPOSE})
    assert len(audit_capture.of()) == before
    audit_capture.assert_chain_ok()


def test_another_owners_job_gets_no_grant_and_the_attempt_is_recorded(
    client: TestClient, audit_capture: AuditCapture
) -> None:
    """Bob gets 404 for Alice's job, as on the snapshot, and the attempt is recorded as denied."""
    job_id = _finished_job(client)
    response = client.post(f"/api/v1/jobs/{job_id}/media-grant", headers={OWNER_HEADER: BOB_OWNER_ID})
    assert response.status_code == 404
    denied = audit_capture.one("query", "media.grant")
    assert denied["outcome"] == "denied"
    assert denied["actor"] == {"user": BOB_OWNER_ID, "via": "edge"}
    assert denied["target"] == {"kind": "job", "ids": [job_id], "count": 1}
    audit_capture.assert_chain_ok()


def test_an_unknown_job_or_a_missing_upload_gets_no_grant(client: TestClient, audit_capture: AuditCapture) -> None:
    """Nothing to play, nothing touched: 404 and no event."""
    job_id = _finished_job(client)
    _manager(client)._jobs[job_id].file_path.unlink()
    before = len(audit_capture.of())
    assert client.post(f"/api/v1/jobs/{'f' * 32}/media-grant").status_code == 404
    assert client.post(f"/api/v1/jobs/{job_id}/media-grant").status_code == 404
    assert len(audit_capture.of()) == before


# --- playing with a grant -----------------------------------------------------------


def test_streams_played_with_a_grant_carry_its_purpose_and_id(client: TestClient, audit_capture: AuditCapture) -> None:
    """Each range is one query, with the grant request's purpose and request id and the grant's id — never the token."""
    job_id = _finished_job(client)
    media_url = _grant(client, job_id, {"X-Audit-Purpose": PURPOSE, "X-Request-ID": GRANT_REQUEST_ID})["media_url"]
    token = _grant_token(media_url)
    player = TestClient(client.app)  # a media element: no identity, no purpose

    full = player.get(media_url, headers={"X-Request-ID": RANGE_REQUEST_ID})
    ranged = player.get(media_url, headers={"Range": "bytes=4-7"})

    assert (full.status_code, full.content) == (200, MEDIA_BYTES)
    assert (ranged.status_code, ranged.content) == (206, MEDIA_BYTES[4:8])
    events = _streams(audit_capture)
    assert [event["criteria"]["ranged"] for event in events] == [False, True]
    grant_ids = {event["criteria"]["grant"] for event in events}
    assert len(grant_ids) == 1
    for event in events:
        assert event["outcome"] == "ok"
        assert event["actor"] == {"user": ALICE_OWNER_ID, "via": "edge"}
        assert event["purpose"] == {"case_ref": "case-0001", "reason": "Review of a reported clip"}
        assert event["request_id"] == GRANT_REQUEST_ID
        assert event["target"] == {"kind": "job", "ids": [job_id], "count": 1}
        assert_no_content(event, token, "clip.wav")
    assert token not in audit_capture.stream.getvalue()
    audit_capture.assert_chain_ok()


def test_a_grant_plays_only_its_own_job(client: TestClient, audit_capture: AuditCapture) -> None:
    """A grant for one job is refused on another's URL, unrecorded."""
    first, second = _finished_job(client), _finished_job(client)
    token = _grant_token(_grant(client, first)["media_url"])
    before = len(audit_capture.of())
    assert client.get(f"/api/v1/jobs/{second}/media", params={"grant": token}).status_code == 404
    assert len(audit_capture.of()) == before


def test_an_unknown_or_expired_grant_is_refused_unrecorded(client: TestClient, audit_capture: AuditCapture) -> None:
    """A grant that names no one is 404 and writes nothing — once expired too."""
    job_id = _finished_job(client)
    now = [1000.0]
    _manager(client).media_grants = MediaGrants(60, clock=lambda: now[0])
    media_url = _grant(client, job_id)["media_url"]
    assert client.get(media_url).status_code == 200
    before = len(audit_capture.of())

    assert client.get(f"/api/v1/jobs/{job_id}/media", params={"grant": "not-a-grant"}).status_code == 404
    now[0] += 60
    assert client.get(media_url).status_code == 404
    assert len(audit_capture.of()) == before
    audit_capture.assert_chain_ok()


def test_deleting_the_job_revokes_its_grants(client: TestClient) -> None:
    """A grant dies with its job."""
    job_id = _finished_job(client)
    media_url = _grant(client, job_id)["media_url"]
    assert client.delete(f"/api/v1/jobs/{job_id}").status_code == 204
    assert client.get(media_url).status_code == 404
    assert len(_manager(client).media_grants) == 0


@pytest.mark.parametrize("query", [pytest.param("", id="neither"), pytest.param("both", id="both")])
def test_media_takes_exactly_one_capability(client: TestClient, query: str) -> None:
    """Neither a token nor a grant opens nothing, and both at once are refused rather than guessed between.

    Args:
        client: Alice's client.
        query: Which capabilities the request presents.
    """
    job_id = _finished_job(client)
    token = parse_qs(urlparse(client.get(f"/api/v1/jobs/{job_id}").json()["result"]["media_url"]).query)["token"][0]
    params: dict[str, str] = {}
    if query:
        params = {"token": token, "grant": _grant_token(_grant(client, job_id)["media_url"])}
    assert client.get(f"/api/v1/jobs/{job_id}/media", params=params).status_code == 404


# --- the required purpose -----------------------------------------------------------


def test_a_grant_needs_the_purpose_a_query_needs(
    monkeypatch: pytest.MonkeyPatch, build: Any, audit_capture: AuditCapture
) -> None:
    """With query required: no grant without a purpose (428, recorded); with one, playback works header-less."""
    monkeypatch.setenv("AUDIT_REQUIRE_PURPOSE", "query,disclose")
    with TestClient(build()) as client:
        client.headers[OWNER_HEADER] = ALICE_OWNER_ID
        job_id = _finished_job(client)
        refused = client.post(f"/api/v1/jobs/{job_id}/media-grant")
        media_url = _grant(client, job_id, {"X-Audit-Purpose": PURPOSE})["media_url"]
        played = TestClient(client.app).get(media_url)
    assert refused.status_code == 428
    assert refused.json() == {"detail": "This request needs a stated purpose."}
    denied = audit_capture.one("query", "media.grant")
    assert denied["outcome"] == "denied"
    assert "target" not in denied
    assert played.status_code == 200
    stream = audit_capture.one("query", "media.stream")
    assert (stream["outcome"], stream["purpose"]["case_ref"]) == ("ok", "case-0001")
    audit_capture.assert_chain_ok()


def test_the_job_token_is_refused_while_query_needs_a_purpose(
    monkeypatch: pytest.MonkeyPatch, build: Any, audit_capture: AuditCapture
) -> None:
    """The per-job token names no purpose, so with query required it is refused (428) and the refusal recorded."""
    monkeypatch.setenv("AUDIT_REQUIRE_PURPOSE", "query")
    with TestClient(build()) as client:
        client.headers[OWNER_HEADER] = ALICE_OWNER_ID
        job_id = _finished_job(client)
        state = _manager(client)._jobs[job_id]
        token_url = f"/api/v1/jobs/{job_id}/media?token={state.media_token}"
        assert TestClient(client.app).get(token_url).status_code == 428
    denied = audit_capture.one("query", "media.stream")
    assert denied["outcome"] == "denied"
    assert denied["actor"] == {"user": ALICE_OWNER_ID, "via": "edge"}
    assert state.media_token not in audit_capture.stream.getvalue()
    audit_capture.assert_chain_ok()


# --- who is recorded ----------------------------------------------------------------


def test_behind_the_gateway_a_grant_plays_only_for_its_holder(
    monkeypatch: pytest.MonkeyPatch, build: Any, audit_capture: AuditCapture
) -> None:
    """With the edge secret, another verified user is refused and recorded; an unverified one is refused unrecorded."""
    monkeypatch.setenv("NEXTEXT_EDGE_SECRET", EDGE_SECRET)
    with TestClient(build()) as client:
        client.headers.update({OWNER_HEADER: ALICE_OWNER_ID, "X-Edge-Secret": EDGE_SECRET})
        job_id = _finished_job(client)
        media_url = _grant(client, job_id, {"X-Audit-Purpose": PURPOSE})["media_url"]
        bob = client.get(media_url, headers={OWNER_HEADER: BOB_OWNER_ID, "X-Request-ID": RANGE_REQUEST_ID})
        direct = TestClient(client.app).get(media_url, headers={OWNER_HEADER: ALICE_OWNER_ID})
        alice = client.get(media_url)
    assert (bob.status_code, direct.status_code, alice.status_code) == (404, 404, 200)
    denied, played = _streams(audit_capture)
    assert denied["outcome"] == "denied"
    assert denied["actor"] == {"user": BOB_OWNER_ID, "via": "edge"}
    assert denied["request_id"] == RANGE_REQUEST_ID
    assert "purpose" not in denied
    assert denied["criteria"]["grant"] == played["criteria"]["grant"]
    assert played["outcome"] == "ok"
    assert played["actor"] == {"user": ALICE_OWNER_ID, "via": "edge"}
    assert played["purpose"]["case_ref"] == "case-0001"
    audit_capture.assert_chain_ok()


def test_without_the_gateway_a_header_less_player_counts_as_the_grant_holder(
    client: TestClient, audit_capture: AuditCapture
) -> None:
    """Without an edge secret, a request with no identity is attributed to whom the grant was issued."""
    job_id = _finished_job(client)
    media_url = _grant(client, job_id, {"X-Audit-Purpose": PURPOSE})["media_url"]
    assert TestClient(client.app).get(media_url).status_code == 200
    stream = audit_capture.one("query", "media.stream")
    assert stream["actor"] == {"user": ALICE_OWNER_ID, "via": "edge"}
    assert stream["purpose"]["reason"] == "Review of a reported clip"
    audit_capture.assert_chain_ok()


def test_without_the_gateway_someone_else_is_recorded_on_the_owners_behalf(
    client: TestClient, audit_capture: AuditCapture
) -> None:
    """Without an edge secret nothing is verified: another identity plays, named as acting on the owner's data."""
    job_id = _finished_job(client)
    media_url = _grant(client, job_id, {"X-Audit-Purpose": PURPOSE})["media_url"]
    assert client.get(media_url, headers={OWNER_HEADER: BOB_OWNER_ID}).status_code == 200
    stream = audit_capture.one("query", "media.stream")
    assert stream["actor"] == {"user": BOB_OWNER_ID, "via": "edge", "on_behalf_of": ALICE_OWNER_ID}
    audit_capture.assert_chain_ok()
