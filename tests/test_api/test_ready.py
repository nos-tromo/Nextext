"""Tests for ``GET /api/v1/ready`` and the Whisper probe behind it.

The probe's own HTTP is faked with a mock transport under the real OpenAI
client, never a network call; the route-level tests inject a stub probe, and a
clock where the cache matters.
"""

from __future__ import annotations

import asyncio
import io
import threading
import time
import wave
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import httpx
import httpx2
import openai
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nextext.api import main as main_module
from nextext.api.main import create_app
from nextext.api.readiness import (
    NOT_CONFIGURED_HINT,
    PROBE_TIMEOUT_S,
    READY_CACHE_TTL_S,
    TranscriptionReadiness,
    WhisperNotConfigured,
    probe_transcription,
    silence_wav,
)
from nextext.api.routes import health as health_module
from nextext.core import transcription as transcription_module
from nextext.core.inference_trace import trace_inference
from tests.audit_helpers import AuditCapture
from tests.test_api.test_skip_outcomes import (
    OWNER_HEADER,
    OWNER_ID,
    _connection_refused_runner,
    _decode_failure_runner,
    _post_job,
    _raising,
    _wait_for_status,
)

WHISPER_URL = "http://whisper.invalid/v1"
ECHOED = "invented upstream detail 0427"


def _connection_error() -> openai.APIConnectionError:
    """Build the error the SDK raises when the endpoint cannot be reached.

    Returns:
        openai.APIConnectionError: A connection failure.
    """
    return openai.APIConnectionError(request=httpx2.Request("POST", f"{WHISPER_URL}/audio/transcriptions"))


def _status_error(status_code: int = 503) -> openai.APIStatusError:
    """Build the error the SDK raises for an HTTP error status, quoting a body.

    Args:
        status_code: The response status.

    Returns:
        openai.APIStatusError: A status error whose message echoes ``ECHOED``.
    """
    request = httpx2.Request("POST", f"{WHISPER_URL}/audio/transcriptions")
    response = httpx2.Response(status_code, request=request, text=f"rejected: {ECHOED}")
    return openai.APIStatusError(f"Error code: {status_code} - {ECHOED}", response=response, body={"error": ECHOED})


class _Clock:
    """A clock that only moves when a test says so."""

    def __init__(self) -> None:
        """Start at an arbitrary instant."""
        self.now = 1000.0

    def __call__(self) -> float:
        """Return the current instant.

        Returns:
            float: Seconds on this clock.
        """
        return self.now


class _Probe:
    """A stand-in probe that counts its calls and plays back scripted outcomes."""

    def __init__(self, *outcomes: Exception | None, hold: threading.Event | None = None) -> None:
        """Script the outcomes.

        Args:
            *outcomes: What each call does, in order: ``None`` succeeds, an
                exception is raised. The last one repeats; no outcome succeeds.
            hold: When set, every call blocks until the event is set.
        """
        self.outcomes = list(outcomes) or [None]
        self.calls = 0
        self.hold = hold
        self.started = threading.Event()

    def __call__(self) -> None:
        """Count the call, wait if told to, then succeed or raise."""
        self.calls += 1
        self.started.set()
        if self.hold is not None:
            assert self.hold.wait(timeout=5.0), "the probe was never released"
        outcome = self.outcomes[min(self.calls, len(self.outcomes)) - 1]
        if outcome is not None:
            raise outcome


class _Pipeline:
    """Stand-in for ``InferencePipeline`` with a fixed health answer."""

    def __init__(self, healthy: bool) -> None:
        """Fix the answer.

        Args:
            healthy: What ``get_health`` returns.
        """
        self.healthy = healthy

    def get_health(self) -> bool:
        """Answer as the real pipeline's check would.

        Returns:
            bool: The fixed answer.
        """
        return self.healthy


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch, audit_capture: AuditCapture) -> FastAPI:
    """Build the app with its audit events captured and logging left alone.

    Args:
        monkeypatch: Pytest patcher.
        audit_capture: Where the app's audit events go.

    Returns:
        FastAPI: A fresh app, whose readiness probe a test replaces.
    """
    monkeypatch.setattr(main_module, "setup_logging", lambda: None)
    return create_app(audit_emitter_factory=audit_capture.factory)


def _arrange(
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    *,
    probe: Callable[[], None],
    inference: bool = True,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    """Install a probe and an inference answer on ``app``.

    Args:
        app: The app under test.
        monkeypatch: Pytest patcher.
        probe: The Whisper probe.
        inference: What the central inference check answers.
        clock: The readiness cache's clock.
    """
    monkeypatch.setattr(health_module, "InferencePipeline", lambda: _Pipeline(inference))
    app.state.readiness = TranscriptionReadiness(probe=probe, clock=clock)


def test_ready_answers_200_when_inference_and_transcription_work(app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> None:
    """Both checks true: ``200`` with status ``ready`` and the package version."""
    probe = _Probe()
    _arrange(app, monkeypatch, probe=probe)

    with TestClient(app) as client:
        response = client.get("/api/v1/ready")

    assert response.status_code == 200
    body = response.json()
    assert body == {"status": "ready", "inference": True, "transcription": True, "version": body["version"]}
    assert isinstance(body["version"], str) and body["version"]
    assert probe.calls == 1


@pytest.mark.parametrize("failure", [_connection_error(), _status_error(503)], ids=["connection", "http-503"])
def test_ready_answers_503_when_whisper_does_not_transcribe_while_health_stays_200(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    """A failing probe makes ``/ready`` ``503``; ``/health`` keeps answering ``200``."""
    _arrange(app, monkeypatch, probe=_Probe(failure))

    with TestClient(app) as client:
        ready = client.get("/api/v1/ready")
        health = client.get("/api/v1/health")

    assert ready.status_code == 503
    body = ready.json()
    assert (body["status"], body["inference"], body["transcription"]) == ("unavailable", True, False)
    assert health.status_code == 200
    assert health.json()["status"] == "ok"


def test_ready_answers_503_when_inference_is_down(app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> None:
    """A central endpoint that does not answer makes ``/ready`` ``503`` with ``inference`` false."""
    _arrange(app, monkeypatch, probe=_Probe(), inference=False)

    with TestClient(app) as client:
        response = client.get("/api/v1/ready")

    assert response.status_code == 503
    body = response.json()
    assert (body["status"], body["inference"], body["transcription"]) == ("unavailable", False, True)


def test_ready_runs_the_inference_check_on_the_pool_health_uses(app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> None:
    """The uncached check goes to anyio's pool, as sync ``/health`` does, not the loop's default executor.

    The default executor is the one the job worker starts its pipeline threads
    on, so a burst of ``/ready`` against a dead router must not queue there.
    """
    threads: dict[str, str] = {}

    def stub(route: str) -> Callable[[], _Pipeline]:
        def build() -> _Pipeline:
            threads[route] = threading.current_thread().name
            return _Pipeline(True)

        return build

    _arrange(app, monkeypatch, probe=_Probe())
    with TestClient(app) as client:
        monkeypatch.setattr(health_module, "InferencePipeline", stub("health"))
        client.get("/api/v1/health")
        monkeypatch.setattr(health_module, "InferencePipeline", stub("ready"))
        client.get("/api/v1/ready")

    assert threads["ready"] == threads["health"]
    assert not threads["ready"].startswith("asyncio_")


def test_ready_survives_an_inference_check_that_raises(app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> None:
    """The route never 500s because the inference check raised."""

    class _Broken:
        def get_health(self) -> bool:
            raise RuntimeError("simulated provider failure")

    _arrange(app, monkeypatch, probe=_Probe())
    monkeypatch.setattr(health_module, "InferencePipeline", _Broken)

    with TestClient(app) as client:
        response = client.get("/api/v1/ready")

    assert response.status_code == 503
    assert response.json()["inference"] is False


def test_ready_probes_once_per_ttl_and_caches_a_failure_too(app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> None:
    """Within 30 s the first result answers, good or bad; after it a new probe runs."""
    clock = _Clock()
    probe = _Probe(_connection_error(), None)
    _arrange(app, monkeypatch, probe=probe, clock=clock)

    with TestClient(app) as client:
        first = client.get("/api/v1/ready")
        clock.now += READY_CACHE_TTL_S - 1
        second = client.get("/api/v1/ready")
        assert probe.calls == 1
        assert (first.status_code, second.status_code) == (503, 503)

        clock.now += 1
        third = client.get("/api/v1/ready")
        fourth = client.get("/api/v1/ready")

    assert probe.calls == 2
    assert (third.status_code, fourth.status_code) == (200, 200)


@pytest.mark.asyncio
async def test_concurrent_requests_share_one_probe(app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> None:
    """Requests that arrive while a probe runs wait for it instead of starting their own."""
    release = threading.Event()
    probe = _Probe(hold=release)
    _arrange(app, monkeypatch, probe=probe)
    transport = httpx.ASGITransport(app=app)

    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            requests = [asyncio.create_task(client.get("/api/v1/ready")) for _ in range(8)]
            assert await asyncio.to_thread(probe.started.wait, 5.0), "no probe started"
            await asyncio.sleep(0.2)
            assert probe.calls == 1, "a second probe started while the first was running"
            release.set()
            responses = await asyncio.gather(*requests)
    finally:
        release.set()

    assert probe.calls == 1
    assert [response.status_code for response in responses] == [200] * 8


@pytest.mark.asyncio
async def test_a_cancelled_request_neither_stops_the_probe_nor_lets_a_second_one_start(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first request gives up mid-probe; the next within the window still shares the one probe."""
    release = threading.Event()
    probe = _Probe(hold=release)
    _arrange(app, monkeypatch, probe=probe)
    transport = httpx.ASGITransport(app=app)

    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            first = asyncio.create_task(client.get("/api/v1/ready"))
            assert await asyncio.to_thread(probe.started.wait, 5.0), "no probe started"
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first

            second = asyncio.create_task(client.get("/api/v1/ready"))
            await asyncio.sleep(0.2)
            assert probe.calls == 1, "a second probe started after the first request gave up"
            release.set()
            answered = await second
            third = await client.get("/api/v1/ready")
    finally:
        release.set()

    assert answered.status_code == 200
    assert third.status_code == 200
    assert probe.calls == 1, "the result of the probe the cancelled request started was lost"


def test_ready_needs_neither_identity_nor_purpose_and_is_not_audited(
    monkeypatch: pytest.MonkeyPatch, audit_capture: AuditCapture
) -> None:
    """With every action needing a purpose and no identity on the request, ``/ready`` still answers."""
    monkeypatch.setenv("NEXTEXT_EDGE_SECRET", "edge-secret-for-tests-only-0123456789abcdef")
    monkeypatch.setenv("AUDIT_REQUIRE_PURPOSE", "query,disclose,collect,ai_call,combine")
    monkeypatch.setattr(main_module, "setup_logging", lambda: None)
    app = create_app(audit_emitter_factory=audit_capture.factory)
    _arrange(app, monkeypatch, probe=_Probe())

    with TestClient(app) as client:
        assert client.get("/api/v1/jobs").status_code == 401
        before = len(audit_capture.events)
        response = client.get("/api/v1/ready")
        after = len(audit_capture.events)

    assert response.status_code == 200
    assert after == before
    audit_capture.assert_chain_ok()


# ---------------------------------------------------------------------------
# A busy Whisper is not a down one
# ---------------------------------------------------------------------------


def _timeout() -> openai.APITimeoutError:
    """Build the error the SDK raises when the probe's request times out.

    Returns:
        openai.APITimeoutError: A timeout.
    """
    return openai.APITimeoutError(request=httpx2.Request("POST", f"{WHISPER_URL}/audio/transcriptions"))


def _after_a_good_answer(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch, second: Exception | None, *, busy: int, good_first: bool = True
) -> tuple[Any, Any]:
    """Run two probes past the cache window: a first, then ``second``, with ``busy`` calls in flight.

    Args:
        app: The app under test.
        monkeypatch: Pytest patcher.
        second: What the second probe raises (``None`` succeeds).
        busy: The number of this process's Whisper requests in flight.
        good_first: Whether the first probe succeeds; otherwise it fails with a connection error.

    Returns:
        tuple[Any, Any]: The first and the second response of ``GET /ready``.
    """
    clock = _Clock()
    probe = _Probe(None if good_first else _connection_error(), second)
    monkeypatch.setattr(health_module, "InferencePipeline", lambda: _Pipeline(True))
    app.state.readiness = TranscriptionReadiness(probe=probe, clock=clock, calls_in_flight=lambda: busy)
    with TestClient(app) as client:
        first = client.get("/api/v1/ready")
        clock.now += READY_CACHE_TTL_S
        answered = client.get("/api/v1/ready")
    assert probe.calls == 2
    return first, answered


def test_a_timeout_while_a_job_transcribes_keeps_the_last_good_answer(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch, log_lines: list[str]
) -> None:
    """The probe queued behind this process's own upload and timed out: still ready, said at debug."""
    first, answered = _after_a_good_answer(app, monkeypatch, _timeout(), busy=1)

    assert (first.status_code, answered.status_code) == (200, 200)
    assert answered.json()["transcription"] is True
    assert any("keeping the last answer" in line for line in log_lines)
    assert not any("readiness probe failed" in line for line in log_lines)


def test_a_timeout_while_busy_with_no_good_answer_yet_is_not_ready(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With nothing good to keep, a timeout is not ready even while a job transcribes."""
    clock = _Clock()
    probe = _Probe(_timeout())
    monkeypatch.setattr(health_module, "InferencePipeline", lambda: _Pipeline(True))
    app.state.readiness = TranscriptionReadiness(probe=probe, clock=clock, calls_in_flight=lambda: 2)

    with TestClient(app) as client:
        response = client.get("/api/v1/ready")

    assert response.status_code == 503
    assert response.json()["transcription"] is False


def test_a_timeout_after_a_failure_while_busy_stays_not_ready(app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only a good answer is kept: after a failure, a busy timeout does not make Whisper ready."""
    first, answered = _after_a_good_answer(app, monkeypatch, _timeout(), busy=1, good_first=False)

    assert (first.status_code, answered.status_code) == (503, 503)


def test_a_timeout_with_nothing_in_flight_is_not_ready(app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> None:
    """A Whisper that is idle as far as this process knows and does not answer is down."""
    first, answered = _after_a_good_answer(app, monkeypatch, _timeout(), busy=0)

    assert first.status_code == 200
    assert answered.status_code == 503
    assert answered.json()["transcription"] is False


@pytest.mark.parametrize(
    "failure",
    [_connection_error(), _status_error(503), _status_error(429), RuntimeError("a bug")],
    ids=["connection", "http-503", "http-429", "other"],
)
def test_anything_but_a_timeout_is_not_ready_whatever_is_in_flight(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    """A refused connection or an error status is an outage, busy or not."""
    first, answered = _after_a_good_answer(app, monkeypatch, failure, busy=3)

    assert first.status_code == 200
    assert answered.status_code == 503


@pytest.mark.asyncio
async def test_a_configuration_that_cannot_work_is_not_ready_whatever_is_in_flight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bad configuration after a good answer is not ready, busy or not."""
    clock = _Clock()
    outcomes: list[Exception | None] = [None, WhisperNotConfigured()]
    calls = 0

    def probe() -> None:
        nonlocal calls
        calls += 1
        outcome = outcomes[calls - 1]
        if outcome is not None:
            raise outcome

    readiness = TranscriptionReadiness(probe=probe, clock=clock, calls_in_flight=lambda: 1)

    assert await readiness.is_ready() is True
    clock.now += READY_CACHE_TTL_S
    assert await readiness.is_ready() is False


@pytest.mark.asyncio
async def test_a_busy_timeout_extends_the_good_answer_only_for_one_window() -> None:
    """The kept answer is re-probed after the next window, and a probe that then answers wins."""
    clock = _Clock()
    probe = _Probe(None, _timeout(), None)
    readiness = TranscriptionReadiness(probe=probe, clock=clock, calls_in_flight=lambda: 1)

    assert await readiness.is_ready() is True
    clock.now += READY_CACHE_TTL_S
    assert await readiness.is_ready() is True
    assert probe.calls == 2
    assert await readiness.is_ready() is True
    assert probe.calls == 2, "the kept answer holds for its window"
    clock.now += READY_CACHE_TTL_S
    assert await readiness.is_ready() is True
    assert probe.calls == 3


# ---------------------------------------------------------------------------
# A job that finds inference down says the cached answer may be wrong
# ---------------------------------------------------------------------------


def _ready_after_a_job_failed_with(
    runner: Any, monkeypatch: pytest.MonkeyPatch, audit_capture: AuditCapture
) -> tuple[int, int, int]:
    """Cache a good ``/ready``, fail one job with ``runner``, ask again, and count the probes.

    Args:
        runner: The pipeline stand-in the job runs.
        monkeypatch: Pytest patcher.
        audit_capture: Where the app's audit events go.

    Returns:
        tuple[int, int, int]: The probe's calls after the first ``/ready``, after a second one
            from the cache, and after the third one asked once the job had failed.
    """
    monkeypatch.setattr(main_module, "setup_logging", lambda: None)
    app = create_app(audit_emitter_factory=audit_capture.factory, pipeline_runner=runner)
    probe = _Probe()
    _arrange(app, monkeypatch, probe=probe)
    client = TestClient(app)
    client.headers[OWNER_HEADER] = OWNER_ID
    with client:
        assert client.get("/api/v1/ready").status_code == 200
        first = probe.calls
        assert client.get("/api/v1/ready").status_code == 200
        cached = probe.calls
        _wait_for_status(client, _post_job(client), "failed")
        assert client.get("/api/v1/ready").status_code == 200
        return first, cached, probe.calls


def test_a_job_that_fails_inference_unavailable_makes_the_next_ready_probe_again(
    monkeypatch: pytest.MonkeyPatch, audit_capture: AuditCapture
) -> None:
    """A cached ``200`` is not served after a job found inference down."""
    assert _ready_after_a_job_failed_with(_connection_refused_runner, monkeypatch, audit_capture) == (1, 1, 2)


def test_a_job_that_fails_for_another_reason_leaves_the_cached_answer_alone(
    monkeypatch: pytest.MonkeyPatch, audit_capture: AuditCapture
) -> None:
    """A bug, a rejected request or a bad upload says nothing about Whisper."""
    for runner in (_raising(lambda: RuntimeError("a bug")), _decode_failure_runner):
        assert _ready_after_a_job_failed_with(runner, monkeypatch, audit_capture) == (1, 1, 1)


def test_a_failing_hook_does_not_fail_the_jobs_bookkeeping(
    monkeypatch: pytest.MonkeyPatch, audit_capture: AuditCapture, log_lines: list[str]
) -> None:
    """The job still ends ``failed`` with its code if the hook raises, and the hook's failure is logged."""
    monkeypatch.setattr(main_module, "setup_logging", lambda: None)
    app = create_app(audit_emitter_factory=audit_capture.factory, pipeline_runner=_connection_refused_runner)

    def broken() -> None:
        raise RuntimeError(ECHOED)

    app.state.readiness = SimpleNamespace(invalidate=broken)
    client = TestClient(app)
    client.headers[OWNER_HEADER] = OWNER_ID
    with client:
        snapshot = _wait_for_status(client, _post_job(client), "failed")

    assert snapshot["error_code"] == "inference_unavailable"
    logged = "\n".join(log_lines)
    assert "The inference-unavailable hook failed: RuntimeError" in logged
    assert ECHOED not in logged


@pytest.mark.asyncio
async def test_invalidate_makes_the_answer_stale_and_a_busy_timeout_cannot_keep_it() -> None:
    """After ``invalidate`` the next request probes, and the invalidated good answer is not kept for a busy timeout."""
    clock = _Clock()
    probe = _Probe(None, None, _timeout())
    readiness = TranscriptionReadiness(probe=probe, clock=clock, calls_in_flight=lambda: 1)

    assert await readiness.is_ready() is True
    assert await readiness.is_ready() is True
    assert probe.calls == 1

    readiness.invalidate()
    assert await readiness.is_ready() is True
    assert probe.calls == 2

    readiness.invalidate()
    clock.now += READY_CACHE_TTL_S
    assert await readiness.is_ready() is False, "a good answer invalidated since must not be kept while busy"
    assert probe.calls == 3


@pytest.mark.asyncio
async def test_a_probe_in_flight_when_invalidated_is_stored_stale() -> None:
    """The request after a probe that started before the invalidation probes again."""
    release = threading.Event()
    probe = _Probe(hold=release)
    readiness = TranscriptionReadiness(probe=probe, clock=_Clock())

    waiting = asyncio.create_task(readiness.is_ready())
    assert await asyncio.to_thread(probe.started.wait, 5.0)
    readiness.invalidate()
    release.set()
    assert await waiting is True
    assert probe.calls == 1

    assert await readiness.is_ready() is True
    assert probe.calls == 2


# ---------------------------------------------------------------------------
# The cache's logging
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_failure_is_logged_once_per_change_of_state_by_class_and_status(log_lines: list[str]) -> None:
    """A failure logs a warning once, however long it lasts; recovery logs once at info; no body is quoted."""
    clock = _Clock()
    probe = _Probe(_status_error(503), _status_error(503), None, None)
    readiness = TranscriptionReadiness(probe=probe, clock=clock)

    results = []
    for _ in range(4):
        results.append(await readiness.is_ready())
        clock.now += READY_CACHE_TTL_S

    assert results == [False, False, True, True]
    logged = "\n".join(log_lines)
    assert logged.count("Transcription readiness probe failed: APIStatusError (HTTP 503)") == 1
    assert logged.count("Transcription readiness probe succeeded again.") == 1
    assert ECHOED not in logged


@pytest.mark.asyncio
async def test_a_first_success_logs_nothing(log_lines: list[str]) -> None:
    """Only a failure and its recovery are news."""
    readiness = TranscriptionReadiness(probe=_Probe(), clock=_Clock())

    assert await readiness.is_ready() is True

    assert not [line for line in log_lines if "readiness" in line]


# ---------------------------------------------------------------------------
# The probe itself
# ---------------------------------------------------------------------------


def _whisper_env(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> None:
    """Point the Whisper settings at an invented vLLM endpoint.

    Args:
        monkeypatch: Pytest patcher.
        **overrides: Environment variables to set on top of the defaults.
    """
    for name in ("WHISPER_API_BASE", "WHISPER_API_KEY", "WHISPER_MODEL"):
        monkeypatch.delenv(name, raising=False)
    env = {"INFERENCE_PROVIDER": "vllm", "OPENAI_API_BASE": WHISPER_URL, "OPENAI_API_KEY": "central-key", **overrides}
    for name, value in env.items():
        monkeypatch.setenv(name, value)


def _fake_whisper_http(
    monkeypatch: pytest.MonkeyPatch, handler: Callable[[httpx2.Request], httpx2.Response]
) -> list[httpx2.Request]:
    """Put a mock transport under the OpenAI client the probe builds.

    Args:
        monkeypatch: Pytest patcher.
        handler: Answers each request the client sends.

    Returns:
        list[httpx2.Request]: Every request the handler saw, in order.
    """
    seen: list[httpx2.Request] = []
    real_client = transcription_module.OpenAI

    def answer(request: httpx2.Request) -> httpx2.Response:
        request.read()
        seen.append(request)
        return handler(request)

    def build(**kwargs: Any) -> openai.OpenAI:
        return real_client(http_client=httpx2.Client(transport=httpx2.MockTransport(answer)), **kwargs)

    monkeypatch.setattr(transcription_module, "OpenAI", build)
    return seen


def test_silence_wav_is_half_a_second_of_16khz_mono_16bit_silence() -> None:
    """The probe's audio is a valid WAV of zero samples."""
    with wave.open(io.BytesIO(silence_wav()), "rb") as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16_000)
        assert wav.getnframes() == 8_000
        assert set(wav.readframes(wav.getnframes())) == {0}


def test_probe_sends_silence_with_the_transcribers_endpoint_key_and_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """One request to ``/audio/transcriptions`` with the Whisper base, key and model, a short timeout."""
    _whisper_env(
        monkeypatch,
        WHISPER_API_BASE="http://dedicated.invalid/v1/",
        WHISPER_API_KEY="whisper-key",
        WHISPER_MODEL="invented/whisper-model",
    )
    seen = _fake_whisper_http(monkeypatch, lambda request: httpx2.Response(200, json={"text": ""}))

    probe_transcription()

    assert len(seen) == 1
    request = seen[0]
    assert request.method == "POST"
    assert str(request.url) == "http://dedicated.invalid/v1/audio/transcriptions"
    assert request.headers["authorization"] == "Bearer whisper-key"
    assert PROBE_TIMEOUT_S == 60.0, "a CPU Whisper needs 10-20 s for even this probe"
    assert request.extensions["timeout"]["read"] == PROBE_TIMEOUT_S
    body = request.content
    assert b'name="model"' in body and b"invented/whisper-model" in body
    assert b'name="response_format"' in body and b"json" in body
    assert b"RIFF" in body and b"WAVE" in body


def test_probe_falls_back_to_the_central_endpoint_key_and_the_provider_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without dedicated settings the probe uses the central pair and the provider's default model."""
    _whisper_env(monkeypatch)
    seen = _fake_whisper_http(monkeypatch, lambda request: httpx2.Response(200, json={"text": ""}))

    probe_transcription()

    assert str(seen[0].url) == f"{WHISPER_URL}/audio/transcriptions"
    assert seen[0].headers["authorization"] == "Bearer central-key"
    assert b"openai/whisper-large-v3" in seen[0].content


@pytest.mark.parametrize("status_code", [429, 500, 503])
def test_probe_fails_on_an_error_status_without_retrying(monkeypatch: pytest.MonkeyPatch, status_code: int) -> None:
    """An HTTP error status raises, and the client asks once: no retries."""
    _whisper_env(monkeypatch)
    seen = _fake_whisper_http(monkeypatch, lambda request: httpx2.Response(status_code, json={"error": "down"}))

    with pytest.raises(openai.APIStatusError):
        probe_transcription()

    assert len(seen) == 1


def test_probe_fails_on_a_connection_error_without_retrying(monkeypatch: pytest.MonkeyPatch) -> None:
    """A refused connection raises, once."""
    _whisper_env(monkeypatch)

    def refuse(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused", request=request)

    seen = _fake_whisper_http(monkeypatch, refuse)

    with pytest.raises(openai.APIConnectionError):
        probe_transcription()

    assert len(seen) == 1


def test_probe_is_not_recorded_as_an_inference_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """The probe never reaches a job's inference tally."""
    _whisper_env(monkeypatch)
    _fake_whisper_http(monkeypatch, lambda request: httpx2.Response(200, json={"text": ""}))

    with trace_inference() as trace:
        probe_transcription()

    assert trace.tallies == []


@pytest.mark.asyncio
async def test_a_configuration_that_cannot_work_is_not_ready_and_logs_a_fixed_hint_once(
    monkeypatch: pytest.MonkeyPatch, log_lines: list[str]
) -> None:
    """Ollama without a Whisper endpoint is not ready; the log names the settings, never the exception's text."""
    _whisper_env(monkeypatch, INFERENCE_PROVIDER="ollama")
    clock = _Clock()
    readiness = TranscriptionReadiness(clock=clock)

    assert await readiness.is_ready() is False
    clock.now += READY_CACHE_TTL_S
    assert await readiness.is_ready() is False

    logged = "\n".join(log_lines)
    assert logged.count(f"Transcription readiness probe failed: {NOT_CONFIGURED_HINT}") == 1
    assert "WHISPER_API_BASE and WHISPER_MODEL" in NOT_CONFIGURED_HINT
    assert "OpenAI-compatible transcription service" not in logged
