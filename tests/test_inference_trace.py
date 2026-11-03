"""Tests for the inference trace: per-task tallies and request correlation.

Every client that sends job content to an inference endpoint reports each
request to the open trace, which the audit log turns into ``ai_call`` events,
and forwards the work's ``X-Request-ID`` — and nothing else about the caller.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from nextext.core import diarization, ner, transcription, vad
from nextext.core.inference_trace import (
    REQUEST_ID_HEADER,
    correlate,
    correlation_headers,
    record_inference,
    reply_usage,
    trace_inference,
)
from nextext.core.openai_cfg import InferencePipeline

REQUEST_ID = "0f4b8c2e-1a2b-4c3d-9e8f-0123456789ab"


def _response(payload: dict[str, Any], status: int = 200) -> httpx.Response:
    """Build an httpx response for a POST to a test endpoint.

    Args:
        payload: The JSON body.
        status: The HTTP status.

    Returns:
        httpx.Response: The response, with a request attached.
    """
    return httpx.Response(status, json=payload, request=httpx.Request("POST", "http://router.invalid/x"))


# --- The trace itself -------------------------------------------------------------


def test_trace_tallies_per_task_in_first_use_order() -> None:
    """Requests are counted per task, failures apart, tasks in first-use order."""
    with trace_inference() as trace:
        record_inference("asr", "whisper-test", ok=True)
        record_inference("chat", "chat-model", ok=True, input_tokens=10, output_tokens=2)
        record_inference("chat", "chat-model", ok=False)
        record_inference("chat", "chat-model", ok=True, input_tokens=5, output_tokens=1)
    tallies = trace.tallies
    assert [tally.task for tally in tallies] == ["asr", "chat"]
    chat = tallies[1]
    assert (chat.requests, chat.failed, chat.succeeded) == (3, 1, 2)
    assert chat.tokens == (15, 3)
    assert chat.model == "chat-model"


def test_tokens_are_given_only_when_every_reply_reported_them() -> None:
    """A partial token sum would understate what was sent, so none is given."""
    with trace_inference() as trace:
        record_inference("chat", "chat-model", ok=True, input_tokens=10, output_tokens=2)
        record_inference("chat", "chat-model", ok=True)
    assert trace.tallies[0].tokens == (None, None)


def test_reports_outside_a_trace_are_dropped() -> None:
    """With no trace open, a report goes nowhere and raises nothing."""
    record_inference("chat", "chat-model", ok=True)
    with trace_inference() as trace:
        pass
    assert trace.tallies == []


def test_correlation_headers_carry_only_the_request_id() -> None:
    """Inside ``correlate`` the headers hold the id; outside, nothing."""
    assert correlation_headers() == {}
    with correlate(REQUEST_ID):
        assert correlation_headers() == {REQUEST_ID_HEADER: REQUEST_ID}
    with correlate(None):
        assert correlation_headers() == {}
    assert correlation_headers() == {}


def test_reply_usage_reads_openai_usage_and_ignores_anything_else() -> None:
    """Token counts come from ``usage``; mocks and missing usage give none."""
    reply = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3))
    assert reply_usage(reply) == (7, 3)
    assert reply_usage(SimpleNamespace(usage=SimpleNamespace(input_tokens=4, output_tokens=1))) == (4, 1)
    assert reply_usage(SimpleNamespace(usage=None)) == (None, None)
    assert reply_usage(MagicMock()) == (None, None)


# --- The clients report and correlate ----------------------------------------------


def _chat_pipeline(monkeypatch: pytest.MonkeyPatch, create: Any) -> InferencePipeline:
    """Build an inference pipeline whose chat endpoint is ``create``.

    Args:
        monkeypatch: Pytest patcher.
        create: Stand-in for ``client.chat.completions.create``.

    Returns:
        InferencePipeline: The pipeline, health check bypassed.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_API_BASE", "http://router.invalid/v1")
    monkeypatch.setenv("TEXT_MODEL", "chat-model")
    pipeline = InferencePipeline()
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(pipeline, "_client", client)
    monkeypatch.setattr(pipeline, "get_health", lambda: True)
    return pipeline


def test_chat_requests_are_reported_with_tokens_and_correlated(monkeypatch: pytest.MonkeyPatch) -> None:
    """A chat call reports its model, served model and tokens, and forwards X-Request-ID only."""
    seen: list[dict[str, Any]] = []

    def create(**kwargs: Any) -> Any:
        seen.append(kwargs)
        message = SimpleNamespace(content=" Answer. ")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=4),
            model="served-model",
        )

    pipeline = _chat_pipeline(monkeypatch, create)
    with trace_inference() as trace, correlate(REQUEST_ID):
        assert pipeline.call_model("Question about case-0001") == "Answer."
    tally = trace.tallies[0]
    assert (tally.task, tally.model, tally.served_model, tally.requests) == ("chat", "chat-model", "served-model", 1)
    assert tally.tokens == (12, 4)
    assert seen[0]["extra_headers"] == {REQUEST_ID_HEADER: REQUEST_ID}


def test_chat_without_correlation_sends_no_extra_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Outside ``correlate`` the request kwargs are unchanged."""
    seen: list[dict[str, Any]] = []

    def create(**kwargs: Any) -> Any:
        seen.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="x"))])

    pipeline = _chat_pipeline(monkeypatch, create)
    pipeline.call_model("hello")
    assert "extra_headers" not in seen[0]


def test_failed_chat_request_is_reported_as_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A request that raises still counts as sent, and as failed."""

    def create(**kwargs: Any) -> Any:
        raise RuntimeError("upstream down")

    pipeline = _chat_pipeline(monkeypatch, create)
    with trace_inference() as trace, pytest.raises(RuntimeError):
        pipeline.call_model("hello")
    assert (trace.tallies[0].requests, trace.tallies[0].failed) == (1, 1)


def test_whisper_request_is_reported_and_correlated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The ASR request reports under task ``asr`` with the Whisper model, carrying X-Request-ID."""
    media = tmp_path / "upload.wav"
    media.write_bytes(b"audio")
    calls: list[dict[str, Any]] = []

    def create(**kwargs: Any) -> Any:
        calls.append(kwargs)
        segment = SimpleNamespace(start=0.0, end=1.0, text="Hello there.", no_speech_prob=0.0)
        return SimpleNamespace(segments=[segment], words=[], language="en", usage=None)

    transcriber = transcription.ExternalWhisperTranscriber(file_path=media, model_id="whisper-test")
    transcriber._client = SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create)))
    monkeypatch.setattr(transcription, "has_speech", lambda path: True)

    def passthrough(file_path: Path, scratch_dir: Path | None = None) -> Any:
        from contextlib import nullcontext

        return nullcontext(file_path)

    monkeypatch.setattr(transcription, "normalize_for_transcription", passthrough)
    with trace_inference() as trace, correlate(REQUEST_ID):
        transcriber.transcription()
    assert [(t.task, t.model, t.requests, t.failed) for t in trace.tallies] == [("asr", "whisper-test", 1, 0)]
    assert calls[0]["extra_headers"] == {REQUEST_ID_HEADER: REQUEST_ID}


@pytest.mark.parametrize(
    ("module", "env", "call", "task", "alias", "ok_payload"),
    [
        pytest.param(
            diarization,
            "DIARIZE_API_BASE",
            lambda path: diarization.diarize_file(path),
            "diarize",
            "diarize",
            {"segments": [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]},
            id="diarize",
        ),
        pytest.param(
            vad, "VAD_API_BASE", lambda path: vad.has_speech(path), "vad", "vad", {"has_speech": True}, id="vad"
        ),
        pytest.param(
            ner,
            "NER_API_BASE",
            lambda path: ner.extract_entities("Alice Example met Bob Example."),
            "ner",
            "gliner",
            {"entities": [{"text": "Alice Example", "label": "person", "score": 0.9}]},
            id="ner",
        ),
    ],
)
def test_service_clients_report_and_correlate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    module: Any,
    env: str,
    call: Any,
    task: str,
    alias: str,
    ok_payload: dict[str, Any],
) -> None:
    """The /diarize, /vad and /gliner clients report each request and forward X-Request-ID.

    Args:
        monkeypatch: Pytest patcher.
        tmp_path: Temporary directory for the media file.
        module: The client module.
        env: Its endpoint variable.
        call: Invokes the client on a path.
        task: The task it reports under.
        alias: The model alias it reports (the endpoint name).
        ok_payload: A successful reply.
    """
    monkeypatch.setenv(env, "http://router.invalid")
    media = tmp_path / "upload.wav"
    media.write_bytes(b"audio")
    sent: list[dict[str, str]] = []
    replies = iter([_response(ok_payload), _response({"detail": "unavailable"}, status=503)])

    def post(url: str, **kwargs: Any) -> httpx.Response:
        sent.append(dict(kwargs.get("headers") or {}))
        return next(replies)

    monkeypatch.setattr(module.httpx, "post", post)
    with trace_inference() as trace, correlate(REQUEST_ID):
        call(media)
        call(media)
    tally = trace.tallies[0]
    assert (tally.task, tally.model, tally.requests, tally.failed) == (task, alias, 2, 1)
    assert all(headers.get(REQUEST_ID_HEADER) == REQUEST_ID for headers in sent)
    assert not any("X-Auth-User" in headers or "X-Audit-Purpose" in headers for headers in sent)


def test_service_clients_send_no_correlation_header_outside_correlate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Without a correlation id, only the bearer token goes out."""
    monkeypatch.setenv("VAD_API_BASE", "http://router.invalid")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    media = tmp_path / "upload.wav"
    media.write_bytes(b"audio")
    sent: list[dict[str, str]] = []

    def post(url: str, **kwargs: Any) -> httpx.Response:
        sent.append(dict(kwargs.get("headers") or {}))
        return _response({"has_speech": True})

    monkeypatch.setattr(vad.httpx, "post", post)
    vad.has_speech(media)
    assert sent == [{"Authorization": "Bearer sk-test"}]
    assert json.dumps(sent).count(REQUEST_ID) == 0
