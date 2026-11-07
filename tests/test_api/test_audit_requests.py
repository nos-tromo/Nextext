"""Tests for the audit events the API writes per request (contract: one per operation).

Each audited route writes exactly one event once the data is fetched and
before it is returned; status polls and progress streams write none; an
attempt on another owner's job is recorded as denied; a stated purpose is
recorded, a malformed one answers 400, a missing required one 428; and an
audit write that fails makes the request fail without its data.
"""

from __future__ import annotations

import hashlib
import io
import json
import threading
import time
from collections.abc import Iterator
from typing import Any, override

import pytest
from fastapi.testclient import TestClient

from nextext.api import main as main_module
from nextext.api.jobs import JobManager, JobState, PushEvent
from nextext.api.main import create_app
from nextext.api.schemas import JobStatus
from nextext.utils.audit_runtime import encode_purpose
from tests.audit_helpers import AuditCapture, assert_no_content

from .conftest import ALICE_OWNER_ID, BOB_OWNER_ID, OWNER_HEADER, stub_pipeline_runner

UPLOAD = b"synthetic-audio-bytes"
UPLOAD_NAME = "Alice Example interview.wav"
REQUEST_ID = "7c9e6679-7425-40de-944b-e07fc1f90ae7"
PURPOSE = encode_purpose("case-0001", "Review of a reported clip")
EDGE_SECRET = "edge-secret-for-tests-only-0123456789abcdef"
DOCINT_TOKEN = "svc-token-docint-for-tests-only-0123456789"


class _BreakableStream(io.StringIO):
    """A buffer whose writes can be made to fail, like a full disk or a closed pipe."""

    broken = False

    @override
    def write(self, s: str) -> int:
        """Write, unless broken.

        Args:
            s: The text.

        Returns:
            int: Characters written.

        Raises:
            OSError: When broken.
        """
        if self.broken:
            raise OSError("audit stream unavailable")
        return super().write(s)


class _Gate:
    """A pipeline runner that waits until released, then runs the stub."""

    def __init__(self) -> None:
        """Start closed."""
        self.event = threading.Event()

    def runner(self, state: JobState, push: PushEvent) -> dict[str, Any]:
        """Block until released, then return the stub result.

        Args:
            state: The job.
            push: The event sink.

        Returns:
            dict[str, Any]: The stub pipeline's result.
        """
        assert self.event.wait(5.0)
        return stub_pipeline_runner(state, push)


@pytest.fixture
def app_factory(monkeypatch: pytest.MonkeyPatch, audit_capture: AuditCapture) -> Any:
    """Return a builder of apps whose audit events go to ``audit_capture``.

    Args:
        monkeypatch: Pytest patcher.
        audit_capture: The capture.

    Returns:
        Any: ``build(runner=stub_pipeline_runner) -> FastAPI``.
    """
    monkeypatch.setattr(main_module, "setup_logging", lambda: None)

    def build(runner: Any = stub_pipeline_runner) -> Any:
        return create_app(audit_emitter_factory=audit_capture.factory, pipeline_runner=runner)

    return build


@pytest.fixture
def client(app_factory: Any) -> Iterator[TestClient]:
    """Yield Alice's client on an app running the stub pipeline.

    Args:
        app_factory: The app builder.

    Yields:
        TestClient: The client.
    """
    with TestClient(app_factory()) as test_client:
        test_client.headers[OWNER_HEADER] = ALICE_OWNER_ID
        yield test_client


def _submit(client: TestClient, headers: dict[str, str] | None = None, **options: Any) -> str:
    """Create a job and return its id.

    Args:
        client: The client.
        headers: Extra request headers.
        **options: Job options.

    Returns:
        str: The job id.
    """
    response = client.post(
        "/api/v1/jobs",
        files={"file": (UPLOAD_NAME, io.BytesIO(UPLOAD), "audio/wav")},
        data={"options": json.dumps({"task": "transcribe", **options})},
        headers=headers or {},
    )
    assert response.status_code == 201, response.text
    return str(response.json()["job_id"])


def _manager(client: TestClient) -> JobManager:
    """Return the app's job manager.

    Args:
        client: The client.

    Returns:
        JobManager: The manager.
    """
    return client.app.state.job_manager  # type: ignore[attr-defined]


def _wait_done(client: TestClient, job_id: str) -> None:
    """Wait until a job has finished, without a request that would be audited.

    Args:
        client: The client.
        job_id: The job.
    """
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        state = _manager(client)._jobs.get(job_id)
        if state is not None and state.status in (JobStatus.COMPLETED, JobStatus.FAILED):
            return
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} did not finish")


def _target(event: dict[str, Any]) -> tuple[str, list[str], int]:
    """Return an event's target kind, ids and count.

    Args:
        event: The event.

    Returns:
        tuple[str, list[str], int]: The target's parts.
    """
    target = event["target"]
    return target["kind"], target.get("ids", []), target["count"]


# --- Chain and operating mode --------------------------------------------------------


def test_startup_records_the_operating_mode(app_factory: Any, audit_capture: AuditCapture) -> None:
    """The chain opens with start and the config event, and closes with shutdown."""
    with TestClient(app_factory()):
        pass
    events = audit_capture.events
    assert [(e["action"], e["operation"]) for e in events[:2]] == [("lifecycle", "start"), ("admin", "config.audit")]
    assert events[1]["actor"] == {"user": None, "via": "system"}
    assert events[1]["criteria"] == {
        "default_identity_enabled": False,
        "edge_secret_enforced": False,
        "require_purpose": [],
        "service_tokens": 0,
        "cli_retention_days": 365,
        "retention_notice_days": 30,
        "faces": "off",
    }
    assert events[1]["service"] == "nextext"
    assert (events[-1]["action"], events[-1]["operation"]) == ("lifecycle", "shutdown")
    audit_capture.assert_chain_ok()


def test_unknown_required_action_stops_startup(monkeypatch: pytest.MonkeyPatch, app_factory: Any) -> None:
    """A misspelt AUDIT_REQUIRE_PURPOSE entry stops the backend instead of switching a requirement off."""
    monkeypatch.setenv("AUDIT_REQUIRE_PURPOSE", "querry")
    with pytest.raises(ValueError, match="querry"), TestClient(app_factory()):
        pass


# --- collect ----------------------------------------------------------------------------


def test_job_creation_is_recorded_as_collect(client: TestClient, audit_capture: AuditCapture) -> None:
    """One collect event: who, which job, how much, what was asked, with purpose and request id."""
    job_id = _submit(client, {"X-Request-ID": REQUEST_ID, "X-Audit-Purpose": PURPOSE}, words=True)
    event = audit_capture.one("collect", "job.create")
    assert event["actor"] == {"user": ALICE_OWNER_ID, "via": "edge"}
    assert _target(event) == ("job", [job_id], 1)
    assert event["request_id"] == REQUEST_ID
    assert event["purpose"] == {"case_ref": "case-0001", "reason": "Review of a reported clip"}
    criteria = event["criteria"]
    assert criteria["files"] == 1
    assert criteria["bytes"] == len(UPLOAD)
    assert criteria["source_sha256"] == hashlib.sha256(UPLOAD).hexdigest()
    assert criteria["analyses"] == ["transcription", "diarization", "word_level"]
    assert criteria["options"]["task"] == "transcribe"
    assert_no_content(event, UPLOAD_NAME, "Alice Example interview")
    _wait_done(client, job_id)
    audit_capture.assert_chain_ok()


def test_job_events_carry_the_creating_request(client: TestClient, audit_capture: AuditCapture) -> None:
    """The background combine event carries the creator's actor, purpose and request id."""
    job_id = _submit(client, {"X-Request-ID": REQUEST_ID, "X-Audit-Purpose": PURPOSE}, summarization=True)
    _wait_done(client, job_id)
    collect = audit_capture.one("collect", "job.create")
    combine = audit_capture.one("combine", "job.combine")
    assert combine["parent_event_id"] == collect["event_id"]
    assert combine["actor"] == collect["actor"]
    assert combine["purpose"] == collect["purpose"]
    assert combine["request_id"] == REQUEST_ID
    assert _target(combine) == ("job", [job_id], 1)
    assert combine["criteria"] == {"analyses": ["diarization", "summary", "transcript"], "speaker_attributed": True}
    assert_no_content(combine, "Hello world.", "A short summary.")
    audit_capture.assert_chain_ok()


def test_rejected_upload_is_recorded_as_a_failed_collection(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, audit_capture: AuditCapture
) -> None:
    """An upload over the size cap is removed and recorded with outcome error."""
    monkeypatch.setenv("NEXTEXT_MAX_UPLOAD_MB", "1")
    response = client.post(
        "/api/v1/jobs",
        files={"file": ("big.wav", io.BytesIO(b"x" * (2 << 20)), "audio/wav")},
        data={"options": json.dumps({"task": "transcribe"})},
    )
    assert response.status_code == 413
    event = audit_capture.one("collect", "job.create")
    assert event["outcome"] == "error"
    assert event["criteria"] == {"failure": "upload_too_large", "files": 1}
    assert "target" not in event
    assert not list(_manager(client).tmp_root.glob("job-*"))
    audit_capture.assert_chain_ok()


# --- query / disclose / delete ------------------------------------------------------


def test_job_list_names_every_job_listed(client: TestClient, audit_capture: AuditCapture) -> None:
    """GET /jobs is one query naming each listed job."""
    first, second = _submit(client), _submit(client)
    _wait_done(client, first)
    _wait_done(client, second)
    assert client.get("/api/v1/jobs", headers={"X-Request-ID": REQUEST_ID}).status_code == 200
    event = audit_capture.one("query", "job.list")
    kind, ids, count = _target(event)
    assert (kind, sorted(ids), count) == ("job", sorted([first, second]), 2)
    assert event["request_id"] == REQUEST_ID
    audit_capture.assert_chain_ok()


def test_finished_snapshot_is_a_query_but_a_status_poll_is_not(app_factory: Any, audit_capture: AuditCapture) -> None:
    """Until the job finishes its snapshot holds no result and is not recorded."""
    gate = _Gate()
    with TestClient(app_factory(gate.runner)) as client:
        client.headers[OWNER_HEADER] = ALICE_OWNER_ID
        job_id = _submit(client)
        assert client.get(f"/api/v1/jobs/{job_id}").json()["result"] is None
        assert audit_capture.of("query") == []
        gate.event.set()
        _wait_done(client, job_id)
        snapshot = client.get(f"/api/v1/jobs/{job_id}").json()
        assert snapshot["result"]["transcript"]
    event = audit_capture.one("query", "job.get")
    assert _target(event) == ("job", [job_id], 1)
    assert_no_content(event, "Hello world.", "token=")
    audit_capture.assert_chain_ok()


def test_cross_owner_attempts_are_recorded_as_denied(client: TestClient, audit_capture: AuditCapture) -> None:
    """Bob gets 404 for Alice's job on every audited route, and each attempt is recorded."""
    job_id = _submit(client)
    _wait_done(client, job_id)
    bob = {OWNER_HEADER: BOB_OWNER_ID}
    assert client.get(f"/api/v1/jobs/{job_id}", headers=bob).status_code == 404
    assert client.get(f"/api/v1/jobs/{job_id}/artifacts/transcript.csv", headers=bob).status_code == 404
    assert client.delete(f"/api/v1/jobs/{job_id}", headers=bob).status_code == 404
    denied = [event for event in audit_capture.of() if event["outcome"] == "denied"]
    assert [(e["action"], e["operation"]) for e in denied] == [
        ("query", "job.get"),
        ("disclose", "artifact.download"),
        ("delete", "job.delete"),
    ]
    for event in denied:
        assert event["actor"] == {"user": BOB_OWNER_ID, "via": "edge"}
        assert _target(event) == ("job", [job_id], 1)
    assert denied[1]["criteria"] == {"artifact": "transcript.csv"}
    audit_capture.assert_chain_ok()


def test_unknown_job_is_not_recorded(client: TestClient, audit_capture: AuditCapture) -> None:
    """A job that does not exist touches no data: 404 and no event."""
    before = len(audit_capture.events)
    assert client.get(f"/api/v1/jobs/{'f' * 32}").status_code == 404
    assert client.delete(f"/api/v1/jobs/{'f' * 32}").status_code == 404
    assert len(audit_capture.events) == before


def test_delete_is_recorded_once_the_job_is_gone(client: TestClient, audit_capture: AuditCapture) -> None:
    """DELETE writes one delete event, by the caller, naming the job."""
    job_id = _submit(client)
    _wait_done(client, job_id)
    response = client.delete(f"/api/v1/jobs/{job_id}", headers={"X-Audit-Purpose": PURPOSE})
    assert response.status_code == 204
    event = audit_capture.one("delete", "job.delete")
    assert event["actor"] == {"user": ALICE_OWNER_ID, "via": "edge"}
    assert _target(event) == ("job", [job_id], 1)
    assert event["criteria"] == {"status": "completed"}
    assert event["purpose"]["case_ref"] == "case-0001"
    assert job_id not in _manager(client)._jobs
    audit_capture.assert_chain_ok()


def test_artifact_download_is_a_disclosure(client: TestClient, audit_capture: AuditCapture) -> None:
    """An artifact download is one disclose to ``download:<format>``; missing ones are not recorded."""
    job_id = _submit(client)
    _wait_done(client, job_id)
    assert client.get(f"/api/v1/jobs/{job_id}/artifacts/transcript.csv").status_code == 200
    assert client.get(f"/api/v1/jobs/{job_id}/artifacts/summary.txt").status_code == 404
    assert client.get(f"/api/v1/jobs/{job_id}/artifacts/not-an-artifact.bin").status_code == 404
    event = audit_capture.one("disclose", "artifact.download")
    assert event["recipient"] == "download:csv"
    assert event["criteria"] == {"artifact": "transcript.csv"}
    assert _target(event) == ("job", [job_id], 1)
    assert_no_content(event, "Hello world.", "Second segment.")
    audit_capture.assert_chain_ok()


def test_batch_download_names_the_jobs_bundled(client: TestClient, audit_capture: AuditCapture) -> None:
    """A batch download is one disclose naming the completed jobs it was built from."""
    first, second = _submit(client), _submit(client)
    _wait_done(client, first)
    _wait_done(client, second)
    assert client.get("/api/v1/jobs/batch/archive.zip").status_code == 200
    event = audit_capture.one("disclose", "artifact.download_batch")
    kind, ids, count = _target(event)
    assert (kind, sorted(ids), count) == ("job", sorted([first, second]), 2)
    assert event["recipient"] == "download:zip"
    audit_capture.assert_chain_ok()


def test_progress_streams_are_not_recorded(client: TestClient, audit_capture: AuditCapture) -> None:
    """The per-job event stream carries no personal data and writes no event."""
    job_id = _submit(client)
    _wait_done(client, job_id)
    before = len(audit_capture.of())
    with client.stream("GET", f"/api/v1/jobs/{job_id}/events") as stream:
        b"".join(stream.iter_bytes())
    assert len(audit_capture.of()) == before


# --- media ----------------------------------------------------------------------------


def _media_url(client: TestClient, job_id: str) -> str:
    """Fetch a finished job's media URL from its snapshot.

    Args:
        client: The owner's client.
        job_id: The job.

    Returns:
        str: The tokenized media URL.
    """
    return str(client.get(f"/api/v1/jobs/{job_id}").json()["result"]["media_url"])


def test_media_stream_is_recorded_without_the_token(client: TestClient, audit_capture: AuditCapture) -> None:
    """Each streamed request is one query; the token never appears in the trail."""
    job_id = _submit(client)
    _wait_done(client, job_id)
    url = _media_url(client, job_id)
    token = url.split("token=", 1)[1]
    assert client.get(url).status_code == 200
    assert client.get(url, headers={"Range": "bytes=0-3"}).status_code == 206
    events = audit_capture.of("query", "media.stream")
    assert [event["criteria"] for event in events] == [{"ranged": False}, {"ranged": True}]
    for event in events:
        assert event["actor"] == {"user": ALICE_OWNER_ID, "via": "edge"}
        assert _target(event) == ("job", [job_id], 1)
    assert token not in audit_capture.stream.getvalue()
    audit_capture.assert_chain_ok()


def test_media_streamed_by_someone_else_names_the_owner(client: TestClient, audit_capture: AuditCapture) -> None:
    """Another identity holding the token is recorded as acting on the owner's data."""
    job_id = _submit(client)
    _wait_done(client, job_id)
    assert client.get(_media_url(client, job_id), headers={OWNER_HEADER: BOB_OWNER_ID}).status_code == 200
    event = audit_capture.one("query", "media.stream")
    assert event["actor"] == {"user": BOB_OWNER_ID, "via": "edge", "on_behalf_of": ALICE_OWNER_ID}


def test_media_without_identity_is_attributed_to_the_creator(client: TestClient, audit_capture: AuditCapture) -> None:
    """Without an edge secret, a header-less media element's request counts as the token's holder, the creator."""
    job_id = _submit(client)
    _wait_done(client, job_id)
    url = _media_url(client, job_id)
    # A second client on the same app, without the owner header (and without
    # running the lifespan again).
    anonymous = TestClient(client.app)
    assert anonymous.get(url).status_code == 200
    event = audit_capture.one("query", "media.stream")
    assert event["actor"] == {"user": ALICE_OWNER_ID, "via": "edge"}


def test_media_without_verified_identity_is_refused_behind_the_gateway(
    monkeypatch: pytest.MonkeyPatch, app_factory: Any, audit_capture: AuditCapture
) -> None:
    """With the edge secret set, a media request that did not come through the gateway gets 404, unrecorded."""
    monkeypatch.setenv("NEXTEXT_EDGE_SECRET", EDGE_SECRET)
    with TestClient(app_factory()) as client:
        client.headers.update({OWNER_HEADER: ALICE_OWNER_ID, "X-Edge-Secret": EDGE_SECRET})
        job_id = _submit(client)
        _wait_done(client, job_id)
        url = _media_url(client, job_id)
        direct = TestClient(client.app)
        assert direct.get(url, headers={OWNER_HEADER: ALICE_OWNER_ID}).status_code == 404
        assert audit_capture.of("query", "media.stream") == []
        assert client.get(url).status_code == 200
    assert audit_capture.one("query", "media.stream")["actor"] == {"user": ALICE_OWNER_ID, "via": "edge"}


# --- purpose ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "header",
    [
        pytest.param("case_ref=a&case_ref=b", id="repeated"),
        pytest.param("case_ref=" + "x" * 300, id="oversized-field"),
        pytest.param("case_ref=a%0Aforged", id="control-character"),
        pytest.param("case_ref=caf\u00e9", id="not-ascii"),
    ],
)
def test_malformed_purpose_is_400_and_unrecorded(client: TestClient, audit_capture: AuditCapture, header: str) -> None:
    """A malformed X-Audit-Purpose answers 400 with a generic message and writes nothing.

    Args:
        client: Alice's client.
        audit_capture: The capture.
        header: The malformed header value.
    """
    before = len(audit_capture.events)
    response = client.get("/api/v1/jobs", headers=[(b"X-Audit-Purpose", header.encode("utf-8"))])
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid audit purpose."}
    assert len(audit_capture.events) == before


def test_required_purpose_refuses_with_428_and_one_denied_event(
    monkeypatch: pytest.MonkeyPatch, app_factory: Any, audit_capture: AuditCapture
) -> None:
    """With query required, a list without a purpose is refused and the refusal recorded; with one it passes."""
    monkeypatch.setenv("AUDIT_REQUIRE_PURPOSE", "query")
    with TestClient(app_factory()) as client:
        client.headers[OWNER_HEADER] = ALICE_OWNER_ID
        refused = client.get("/api/v1/jobs", headers={"X-Request-ID": REQUEST_ID})
        allowed = client.get("/api/v1/jobs", headers={"X-Audit-Purpose": PURPOSE})
    assert refused.status_code == 428
    assert refused.json() == {"detail": "This request needs a stated purpose."}
    assert allowed.status_code == 200
    denied, ok = audit_capture.of("query", "job.list")
    assert (denied["outcome"], ok["outcome"]) == ("denied", "ok")
    assert "target" not in denied
    assert denied["request_id"] == REQUEST_ID
    assert ok["purpose"] == {"case_ref": "case-0001", "reason": "Review of a reported clip"}
    assert audit_capture.events[1]["criteria"]["require_purpose"] == ["query"]
    audit_capture.assert_chain_ok()


def test_required_model_call_purpose_gates_job_creation(
    monkeypatch: pytest.MonkeyPatch, app_factory: Any, audit_capture: AuditCapture
) -> None:
    """A purpose required for ai_call is required to create a job, whose model calls need it."""
    monkeypatch.setenv("AUDIT_REQUIRE_PURPOSE", "ai_call")
    with TestClient(app_factory()) as client:
        client.headers[OWNER_HEADER] = ALICE_OWNER_ID
        refused = client.post(
            "/api/v1/jobs",
            files={"file": ("clip.wav", io.BytesIO(UPLOAD), "audio/wav")},
            data={"options": json.dumps({"task": "transcribe"})},
        )
        assert refused.status_code == 428
        assert _manager(client)._jobs == {}
        job_id = _submit(client, {"X-Audit-Purpose": PURPOSE})
        _wait_done(client, job_id)
    denied, ok = audit_capture.of("collect", "job.create")
    assert (denied["outcome"], ok["outcome"]) == ("denied", "ok")
    audit_capture.assert_chain_ok()


def test_status_poll_needs_no_purpose_but_the_result_does(
    monkeypatch: pytest.MonkeyPatch, app_factory: Any, audit_capture: AuditCapture
) -> None:
    """With query required, polling a running job works; reading the finished result needs a purpose."""
    monkeypatch.setenv("AUDIT_REQUIRE_PURPOSE", "query")
    gate = _Gate()
    with TestClient(app_factory(gate.runner)) as client:
        client.headers[OWNER_HEADER] = ALICE_OWNER_ID
        job_id = _submit(client)
        assert client.get(f"/api/v1/jobs/{job_id}").status_code == 200
        gate.event.set()
        _wait_done(client, job_id)
        assert client.get(f"/api/v1/jobs/{job_id}").status_code == 428
        assert client.get(f"/api/v1/jobs/{job_id}", headers={"X-Audit-Purpose": PURPOSE}).status_code == 200
    assert [event["outcome"] for event in audit_capture.of("query", "job.get")] == ["denied", "ok"]


def test_no_purpose_is_needed_by_default(client: TestClient, audit_capture: AuditCapture) -> None:
    """With AUDIT_REQUIRE_PURPOSE empty nothing is refused, and events carry no purpose."""
    job_id = _submit(client)
    _wait_done(client, job_id)
    assert client.get(f"/api/v1/jobs/{job_id}").status_code == 200
    assert all("purpose" not in event for event in audit_capture.of())


# --- actor mapping ----------------------------------------------------------------------


def test_service_token_caller_is_a_service_actor(
    monkeypatch: pytest.MonkeyPatch, app_factory: Any, audit_capture: AuditCapture
) -> None:
    """Docint's token resolves to ``Actor("service:docint", "service")``, whatever header it also sends."""
    monkeypatch.setenv("NEXTEXT_SERVICE_TOKENS", f"docint={DOCINT_TOKEN}")
    with TestClient(app_factory()) as client:
        client.headers.update({"Authorization": f"Bearer {DOCINT_TOKEN}", OWNER_HEADER: "docint"})
        _submit(client)
    assert audit_capture.one("collect", "job.create")["actor"] == {"user": "service:docint", "via": "service"}


def test_default_identity_is_a_system_actor(
    monkeypatch: pytest.MonkeyPatch, app_factory: Any, audit_capture: AuditCapture
) -> None:
    """The dev fallback identity is recorded via ``system``."""
    monkeypatch.setenv("NEXTEXT_DEFAULT_IDENTITY", "dev-example")
    with TestClient(app_factory()) as client:
        _submit(client)
    assert audit_capture.one("collect", "job.create")["actor"] == {"user": "dev-example", "via": "system"}
    assert audit_capture.events[1]["criteria"]["default_identity_enabled"] is True


def test_unverified_header_is_unauthenticated_and_unrecorded(
    monkeypatch: pytest.MonkeyPatch, app_factory: Any, audit_capture: AuditCapture
) -> None:
    """With the edge secret set, a header without it is 401 and writes no event; with it, via edge."""
    monkeypatch.setenv("NEXTEXT_EDGE_SECRET", EDGE_SECRET)
    with TestClient(app_factory()) as client:
        assert client.get("/api/v1/jobs", headers={OWNER_HEADER: "forged-owner"}).status_code == 401
        assert audit_capture.of() == [audit_capture.events[1]]
        verified = client.get("/api/v1/jobs", headers={OWNER_HEADER: "alice", "X-Edge-Secret": EDGE_SECRET})
        assert verified.status_code == 200
    assert audit_capture.one("query", "job.list")["actor"] == {"user": "alice", "via": "edge"}
    assert audit_capture.events[1]["criteria"]["edge_secret_enforced"] is True


# --- fail closed ------------------------------------------------------------------------


@pytest.fixture
def breakable(audit_capture: AuditCapture) -> _BreakableStream:
    """Swap the capture's buffer for one whose writes can be made to fail.

    Args:
        audit_capture: The capture.

    Returns:
        _BreakableStream: The buffer.
    """
    stream = _BreakableStream()
    audit_capture.stream = stream
    return stream


def test_read_fails_closed_when_its_event_cannot_be_written(
    breakable: _BreakableStream, client: TestClient, audit_capture: AuditCapture
) -> None:
    """A result that cannot be recorded is not returned: generic 500, no data."""
    job_id = _submit(client)
    _wait_done(client, job_id)
    failing = TestClient(client.app, raise_server_exceptions=False)
    failing.headers[OWNER_HEADER] = ALICE_OWNER_ID
    breakable.broken = True
    snapshot = failing.get(f"/api/v1/jobs/{job_id}")
    download = failing.get(f"/api/v1/jobs/{job_id}/artifacts/transcript.csv")
    breakable.broken = False
    for response in (snapshot, download):
        assert response.status_code == 500
        assert response.json() == {"detail": "Internal server error."}
        assert "Hello world." not in response.text
    audit_capture.assert_chain_ok()


def test_job_creation_fails_closed_when_collect_cannot_be_written(
    breakable: _BreakableStream, client: TestClient, audit_capture: AuditCapture
) -> None:
    """An upload whose collection cannot be recorded is removed, and no job exists."""
    failing = TestClient(client.app, raise_server_exceptions=False)
    failing.headers[OWNER_HEADER] = ALICE_OWNER_ID
    breakable.broken = True
    response = failing.post(
        "/api/v1/jobs",
        files={"file": ("clip.wav", io.BytesIO(UPLOAD), "audio/wav")},
        data={"options": json.dumps({"task": "transcribe"})},
    )
    breakable.broken = False
    assert response.status_code == 500
    assert _manager(client)._jobs == {}
    assert not list(_manager(client).tmp_root.glob("job-*"))
    audit_capture.assert_chain_ok()


def test_job_fails_when_its_events_cannot_be_written(
    breakable: _BreakableStream, app_factory: Any, audit_capture: AuditCapture
) -> None:
    """A job whose combination cannot be recorded fails and offers no result."""
    gate = _Gate()
    with TestClient(app_factory(gate.runner)) as client:
        client.headers[OWNER_HEADER] = ALICE_OWNER_ID
        job_id = _submit(client)
        breakable.broken = True
        gate.event.set()
        _wait_done(client, job_id)
        breakable.broken = False
        snapshot = client.get(f"/api/v1/jobs/{job_id}").json()
    assert snapshot["status"] == "failed"
    assert snapshot["result"] is None
    assert audit_capture.of("combine") == []
    audit_capture.assert_chain_ok()
