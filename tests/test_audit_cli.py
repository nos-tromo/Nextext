"""Tests for the audit chain a ``nextext-cli`` run writes (``nextext.cli.run_cli``).

Each run is its own chain, attributed to ``AUDIT_CLI_USER`` via ``cli``: the
file's intake, every model call, the joined result and every output file. A
missing required purpose exits 2 with one recorded refusal; malformed audit
settings exit 2 with nothing recorded; a failed audit write stops the run.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any, override

import pandas as pd
import pytest

from nextext import cli
from nextext.core.inference_trace import record_inference
from nextext.pipeline import TranscriptionOutcome
from nextext.utils.audit_event import AuditEmitter, AuditWriteError, verify_chain
from tests.audit_helpers import AuditCapture, assert_no_content

MEDIA_NAME = "Alice Example interview.wav"
MEDIA = b"synthetic-cli-audio"
TRANSCRIPT = pd.DataFrame(
    {
        "start": ["0:00:00", "0:00:02"],
        "end": ["0:00:02", "0:00:04"],
        "speaker": ["Speaker 1", "Speaker 2"],
        "text": ["Alice Example opened.", "Bob Example replied."],
    }
)


class _HealthyInference:
    """Stand-in for ``InferencePipeline`` that is always reachable."""

    def get_health(self) -> bool:
        """Report the provider as reachable.

        Returns:
            bool: Always ``True``.
        """
        return True


@pytest.fixture
def media(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Write a synthetic input file and run the CLI from its directory.

    Args:
        monkeypatch: Pytest patcher.
        tmp_path: Temporary directory.

    Returns:
        Path: The input file.
    """
    monkeypatch.chdir(tmp_path)
    path = tmp_path / MEDIA_NAME
    path.write_bytes(MEDIA)
    return path


@pytest.fixture
def stub_models(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace every model-backed step of the CLI with a fake that reports its requests.

    Args:
        monkeypatch: Pytest patcher.

    Returns:
        list[str]: The names of the steps that ran, in order.
    """
    ran: list[str] = []

    def transcription(**kwargs: Any) -> TranscriptionOutcome:
        ran.append("transcription")
        record_inference("vad", "vad", ok=True)
        record_inference("asr", "whisper-test", ok=True)
        record_inference("diarize", "diarize", ok=True)
        return TranscriptionOutcome(transcript=TRANSCRIPT.copy(), src_lang="en")

    def summarization(**kwargs: Any) -> str:
        ran.append("summary")
        record_inference("chat", "chat-model", ok=True, input_tokens=40, output_tokens=9)
        return "An invented summary."

    def hate_speech(**kwargs: Any) -> list[dict[str, Any]]:
        ran.append("hate_speech")
        record_inference("chat", "chat-model", ok=True)
        return [{"hate_speech": True, "category": "other", "confidence": "low", "reason": "r", "text": "x"}]

    monkeypatch.setattr(cli, "transcription_pipeline", transcription)
    monkeypatch.setattr(cli, "summarization_pipeline", summarization)
    monkeypatch.setattr(cli, "hate_speech_pipeline", hate_speech)
    monkeypatch.setattr(cli, "frame_hate_speech_pipeline", lambda captions, **kwargs: [])
    monkeypatch.setattr(cli, "InferencePipeline", _HealthyInference)
    return ran


def test_a_run_writes_its_own_chain(
    monkeypatch: pytest.MonkeyPatch, media: Path, stub_models: list[str], audit_capture: AuditCapture
) -> None:
    """Intake, model calls, combination and outputs — all by the CLI operator, with the stated purpose."""
    monkeypatch.setenv("AUDIT_CLI_USER", "operator-example")
    monkeypatch.setenv("AUDIT_CASE_REF", "case-0002")
    monkeypatch.setenv("AUDIT_REASON", "Batch review of invented clips")
    args = cli.parse_arguments(["-f", str(media), "-sum", "-hs"])

    assert cli.run_cli(args, emitter_factory=audit_capture.factory) == 0

    events = audit_capture.events
    assert (events[0]["operation"], events[-1]["operation"]) == ("start", "shutdown")
    config = audit_capture.one("admin", "config.audit")
    assert config["criteria"] == {"require_purpose": []}
    collect = audit_capture.one("collect", "job.create")
    assert collect["actor"] == {"user": "operator-example", "via": "cli"}
    assert collect["purpose"] == {"case_ref": "case-0002", "reason": "Batch review of invented clips"}
    assert collect["criteria"]["bytes"] == len(MEDIA)
    assert collect["criteria"]["source_sha256"] == hashlib.sha256(MEDIA).hexdigest()
    assert collect["criteria"]["analyses"] == ["transcription", "diarization", "summary", "hate_speech"]
    run_id = collect["target"]["ids"][0]
    assert [event["operation"] for event in audit_capture.of("ai_call")] == [
        "ai.vad",
        "ai.transcribe",
        "ai.diarize",
        "ai.summarize",
        "ai.hate_speech",
    ]
    combine = audit_capture.one("combine", "job.combine")
    assert combine["criteria"]["analyses"] == ["diarization", "hate_speech", "summary", "transcript"]
    outputs = [event["recipient"] for event in audit_capture.of("disclose", "output.write")]
    assert outputs == ["file:summary", "file:hate_speech", "file:transcript"]
    for event in audit_capture.of():
        if event["action"] != "admin":
            assert event["target"]["ids"] == [run_id]
            assert event["actor"]["via"] == "cli"
            assert event.get("parent_event_id") in (None, collect["event_id"])
        assert_no_content(event, MEDIA_NAME, "Alice Example interview", str(media.parent), "An invented summary.")
    assert stub_models == ["transcription", "summary", "hate_speech"]
    audit_capture.assert_chain_ok()


def test_a_run_without_a_cli_user_has_no_user(media: Path, stub_models: list[str], audit_capture: AuditCapture) -> None:
    """Outside ``make cli`` nobody is named: the actor's user is null, via cli."""
    assert cli.run_cli(cli.parse_arguments(["-f", str(media)]), emitter_factory=audit_capture.factory) == 0
    assert audit_capture.one("collect", "job.create")["actor"] == {"user": None, "via": "cli"}
    assert "purpose" not in audit_capture.one("collect", "job.create")


def test_a_required_purpose_refuses_the_run_with_exit_2(
    monkeypatch: pytest.MonkeyPatch,
    media: Path,
    stub_models: list[str],
    audit_capture: AuditCapture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """With a purpose required and none stated: one denied event, a one-line message, exit 2, no processing."""
    monkeypatch.setenv("AUDIT_REQUIRE_PURPOSE", "disclose")
    assert cli.run_cli(cli.parse_arguments(["-f", str(media)]), emitter_factory=audit_capture.factory) == 2
    denied = audit_capture.one("collect", "job.create")
    assert denied["outcome"] == "denied"
    assert "target" not in denied
    assert stub_models == []
    err = capsys.readouterr().err.strip().splitlines()
    assert err == ["nextext-cli: this run needs a stated purpose: set AUDIT_CASE_REF and/or AUDIT_REASON."]
    audit_capture.assert_chain_ok()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        pytest.param("AUDIT_CASE_REF", "case\u0007bell", id="control-character"),
        pytest.param("AUDIT_REASON", "x" * 300, id="oversized"),
        pytest.param("AUDIT_REQUIRE_PURPOSE", "collect,querry", id="unknown-action"),
    ],
)
def test_malformed_audit_settings_exit_2_and_record_nothing(
    monkeypatch: pytest.MonkeyPatch,
    media: Path,
    stub_models: list[str],
    audit_capture: AuditCapture,
    capsys: pytest.CaptureFixture[str],
    name: str,
    value: str,
) -> None:
    """A malformed purpose or requirement stops the run before its chain starts.

    Args:
        monkeypatch: Pytest patcher.
        media: The input file.
        stub_models: The model fakes.
        audit_capture: The capture.
        capsys: Captured output.
        name: The variable to set.
        value: Its malformed value.
    """
    monkeypatch.setenv(name, value)
    assert cli.run_cli(cli.parse_arguments(["-f", str(media)]), emitter_factory=audit_capture.factory) == 2
    assert audit_capture.emitters == []
    assert stub_models == []
    message = capsys.readouterr().err.strip()
    assert message.startswith("nextext-cli: ")
    assert value not in message or name == "AUDIT_REQUIRE_PURPOSE"


def test_a_failed_audit_write_stops_the_run(media: Path, stub_models: list[str]) -> None:
    """When the chain cannot be written, the run raises (non-zero exit) and processes nothing further."""

    class _FailsAfterTwoLines(io.StringIO):
        healed = False

        @override
        def write(self, s: str) -> int:
            if not self.healed and self.getvalue().count("\n") >= 2:
                raise OSError("closed pipe")
            return super().write(s)

    stream = _FailsAfterTwoLines()
    made: list[AuditEmitter] = []

    def factory(output: str | None) -> AuditEmitter:
        made.append(AuditEmitter("nextext", "0", stream=stream, heartbeat_interval_s=0))
        return made[-1]

    with pytest.raises(AuditWriteError):
        cli.run_cli(cli.parse_arguments(["-f", str(media)]), emitter_factory=factory)
    assert stub_models == []
    # Let the chain end cleanly, so nothing retries its shutdown at exit.
    stream.healed = True
    made[0].close()


def test_main_writes_the_chain_to_stdout_and_nothing_else(
    monkeypatch: pytest.MonkeyPatch, media: Path, stub_models: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    """Run as a command, the CLI's stdout carries its audit chain alone."""
    monkeypatch.setattr(sys, "argv", ["nextext-cli", "-f", str(media)])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 0
    lines = capsys.readouterr().out.splitlines()
    events = [json.loads(line) for line in lines]
    assert all(event["stream"] == "audit" for event in events)
    assert verify_chain(events).ok
