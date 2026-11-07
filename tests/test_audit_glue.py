"""Unit tests for the audit glue shared by the API and the CLI (``nextext.audit``)."""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import override

import pytest

from nextext import audit as audit_module
from nextext.audit import (
    FACE_STAGE,
    SERVICE,
    TRANSCRIPTION_STAGE,
    AuditSink,
    build_emitter,
    emit_with_ids,
    joined_analyses,
    service_version,
)
from nextext.core.inference_trace import record_inference
from nextext.utils.audit_event import Actor, AuditEmitter
from nextext.utils.audit_runtime import ContainerAuditStream
from tests.audit_helpers import AuditCapture, assert_no_content, capturing_job_audit


def test_long_id_lists_are_continued_never_cut() -> None:
    """450 ids: the first event holds 200 and counts all; two continuations point at it."""
    capture = AuditCapture()
    emitter = capture.started()
    ids = [f"{index:032x}" for index in range(450)]

    primary = emit_with_ids(emitter, "query", "job.list", Actor("alice.example", "edge"), kind="job", ids=ids)

    events = capture.of()
    assert [event["operation"] for event in events] == ["job.list", "job.list.ids", "job.list.ids"]
    assert events[0]["target"]["count"] == 450
    assert events[0]["target"]["ids"] == ids[:200]
    assert "ids_truncated" not in events[0]["target"]
    assert [event["target"]["count"] for event in events[1:]] == [200, 50]
    assert all(event["parent_event_id"] == primary["event_id"] for event in events[1:])
    assert events[1]["target"]["ids"] + events[2]["target"]["ids"] == ids[200:]
    capture.assert_chain_ok()


def test_empty_id_list_is_one_event_counting_zero() -> None:
    """A query that found nothing is still recorded, with count 0."""
    capture = AuditCapture()
    emit_with_ids(capture.started(), "query", "job.list", Actor("alice.example", "edge"), kind="job", ids=[])
    assert capture.one("query", "job.list")["target"] == {"kind": "job", "count": 0}


def test_sealed_sink_refuses_events_but_the_emitter_still_writes() -> None:
    """After seal() job events raise; the owner writes its last events on the emitter itself."""
    capture = AuditCapture()
    sink = AuditSink(capture.started())
    sink.emit("admin", "probe", Actor(None, "system"))
    sink.seal()
    with pytest.raises(RuntimeError):
        sink.emit("admin", "probe", Actor(None, "system"))
    sink.emitter.emit("delete", "job.purge", Actor(None, "system"))
    assert [event["operation"] for event in capture.of()] == ["probe", "job.purge"]


def test_a_task_without_a_spec_is_still_recorded() -> None:
    """An inference task a stage table does not name is recorded generically, never dropped."""
    audit, capture = capturing_job_audit()
    with audit.stage(TRANSCRIPTION_STAGE):
        record_inference("embed", "embed-model", ok=True)
    event = capture.one("ai_call", "ai.embed")
    assert event["recipient"] == "inference:embed"
    assert event["ai"]["model_alias"] == "embed-model"


def test_a_stage_without_requests_writes_nothing() -> None:
    """A stage that sent nothing (e.g. VAD switched off, no diarization) records no transfer."""
    audit, capture = capturing_job_audit()
    with audit.stage(TRANSCRIPTION_STAGE) as notes:
        notes.result("asr", text="never sent", segments=0)
    assert capture.of() == []


def test_combination_needs_two_analyses() -> None:
    """A lone transcript combines nothing; speaker labels plus the transcript do."""
    audit, capture = capturing_job_audit()
    assert audit.combine(["transcript"], speaker_attributed=False) is None
    assert audit.combine(["transcript", "diarization"], speaker_attributed=True) is not None
    assert capture.one("combine", "job.combine")["criteria"] == {
        "analyses": ["diarization", "transcript"],
        "speaker_attributed": True,
    }


def test_joined_analyses_follow_the_transcript_columns() -> None:
    """Speaker and translation columns mean diarization and translation were joined."""
    analyses, speaker = joined_analyses(
        transcript_columns=["start", "end", "speaker", "text", "translation"],
        word_counts=True,
        named_entities=False,
        summary=True,
        frame_captions=False,
        faces=False,
        hate_speech=True,
    )
    assert analyses == ["transcript", "diarization", "translation", "word_counts", "summary", "hate_speech"]
    assert speaker is True


def test_joined_analyses_lists_faces() -> None:
    """The face stage is listed after the frame captions whenever it ran."""
    analyses, speaker = joined_analyses(
        transcript_columns=[],
        word_counts=False,
        named_entities=False,
        summary=False,
        frame_captions=True,
        faces=True,
        hate_speech=False,
    )
    assert analyses == ["frame_captions", "faces"]
    assert speaker is False


def test_face_stage_records_frames_and_tracks_only() -> None:
    """Detection and grouping are two transfers of the upload, recorded by what was judged."""
    audit, capture = capturing_job_audit("f" * 32, "alice.example")
    collect = audit.collect({"files": 1})

    with audit.stage(FACE_STAGE) as notes:
        record_inference("face_detect", "faces", ok=True)
        record_inference("face_detect", "faces", ok=False)
        record_inference("face_group", "faces", ok=True)
        notes.result("face_detect", frames=12)
        notes.result("face_group", tracks=3)

    detect = capture.one("ai_call", "ai.detect_faces")
    group = capture.one("ai_call", "ai.group_faces")
    assert (detect["recipient"], group["recipient"]) == ("inference:faces", "inference:faces")
    assert detect["criteria"] == {"requests": 2, "failed_requests": 1, "frames": 12}
    assert group["criteria"] == {"requests": 1, "tracks": 3}
    assert detect["ai"]["sources"] == [collect["event_id"]]
    assert group["ai"]["sources"] == [collect["event_id"], detect["event_id"]]
    assert detect["ai"]["model_alias"] == "faces"
    for event in (detect, group):
        assert "output_sha256" not in event["ai"]
        assert_no_content(event, "person", "Person", "persons", "faces_found", "sha256:")
    capture.assert_chain_ok()


def test_build_emitter_honours_the_output_mode(tmp_path: Path) -> None:
    """Stdout leaves the stream to the library; container opens the locked container stream; others fail."""
    plain = build_emitter(None)
    assert (plain.service, plain.service_version) == (SERVICE, service_version())
    assert plain._stream is None
    with pytest.raises(ValueError, match="AUDIT_OUTPUT"):
        build_emitter("syslog")


def test_container_mode_writes_whole_lines_to_the_target(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """In container mode every line goes to the container's stdout, here a file standing in for it."""
    target = tmp_path / "container-stdout"
    target.write_text("")
    lock = tmp_path / "audit-stream.lock"

    def opener(output: str | None) -> ContainerAuditStream:
        assert output == "container"
        return ContainerAuditStream(str(target), str(lock))

    monkeypatch.setattr(audit_module, "open_audit_stream", opener)
    emitter = build_emitter("container", heartbeat_interval_s=0)
    emitter.start()
    emitter.close()
    lines = target.read_text().splitlines()
    assert [line.count('"stream":"audit"') for line in lines] == [1, 1]
    assert oct(os.stat(lock).st_mode & 0o777) == "0o600"


def test_emitter_write_failure_surfaces() -> None:
    """A broken stream raises, which callers let fail the operation."""

    class _Closed(io.StringIO):
        @override
        def write(self, s: str) -> int:
            raise OSError("closed pipe")

    emitter = AuditEmitter(SERVICE, "0", stream=_Closed(), heartbeat_interval_s=0)
    with pytest.raises(Exception, match="could not be written"):
        emitter.start()
