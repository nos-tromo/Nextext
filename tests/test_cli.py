"""Tests for the CLI's transcript write sites (``nextext.cli._run_main``).

Guards that both places the pipeline saves the transcript — the no-speech guard
and the final write — route through ``FileProcessor.write_transcript_output``
(which emits the readable ``.txt`` blocks) rather than the generic
``write_file_output``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from loguru import logger

from nextext import cli
from nextext.core.outcomes import SkipReason
from nextext.pipeline import TranscriptionOutcome


class _SpyProcessor:
    """Stand-in for ``FileProcessor`` recording how the transcript was saved."""

    def __init__(self) -> None:
        self.transcript_writes: list[pd.DataFrame] = []
        self.file_output_labels: list[str] = []

    def write_transcript_output(self, data: pd.DataFrame) -> None:
        self.transcript_writes.append(data)

    def write_file_output(self, data: Any, label: str, target_language: str = "") -> Any:
        self.file_output_labels.append(label)
        return data


def _args(file_path: Path, **overrides: Any) -> argparse.Namespace:
    """Build a minimal transcribe-task ``argparse.Namespace`` for ``_run_main``."""
    base: dict[str, Any] = {
        "file_path": file_path,
        "src_lang": "en",
        "trg_lang": "en",
        "task": "transcribe",
        "diarize": True,
        "words": False,
        "summarize": False,
        "hate_speech": False,
        "emit_docint_jsonl": None,
        "force_docint_jsonl": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_run_main_saves_transcript_via_write_transcript_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The normal transcribe path saves the transcript through write_transcript_output."""
    created: list[_SpyProcessor] = []

    def _make(*args: Any, **kwargs: Any) -> _SpyProcessor:
        processor = _SpyProcessor()
        created.append(processor)
        return processor

    monkeypatch.setattr(cli, "FileProcessor", _make)
    df = pd.DataFrame({"start": ["0:00:00"], "end": ["0:00:02"], "text": ["Hello."]})
    monkeypatch.setattr(
        cli, "transcription_pipeline", lambda **kwargs: TranscriptionOutcome(transcript=df, src_lang="en")
    )

    cli._run_main(_args(tmp_path / "clip.wav"))

    (processor,) = created
    assert len(processor.transcript_writes) == 1
    # The transcript must NOT be saved via the generic write_file_output.
    assert "transcript" not in processor.file_output_labels


def test_run_main_no_speech_saves_via_write_transcript_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The no-speech guard also saves the (empty) transcript through write_transcript_output."""
    created: list[_SpyProcessor] = []

    def _make(*args: Any, **kwargs: Any) -> _SpyProcessor:
        processor = _SpyProcessor()
        created.append(processor)
        return processor

    monkeypatch.setattr(cli, "FileProcessor", _make)
    empty = pd.DataFrame(columns=pd.Index(["start", "end", "text"]))
    monkeypatch.setattr(
        cli,
        "transcription_pipeline",
        lambda **kwargs: TranscriptionOutcome(transcript=empty, src_lang="en", skip_reason="vad_no_speech"),
    )

    cli._run_main(_args(tmp_path / "clip.wav"))

    (processor,) = created
    assert len(processor.transcript_writes) == 1
    assert "transcript" not in processor.file_output_labels


def test_cli_diarize_defaults_on_and_can_be_disabled() -> None:
    """--diarize defaults True; --no-diarize turns it off."""
    from nextext.cli import parse_arguments

    assert parse_arguments(["-f", "x.wav"]).diarize is True
    assert parse_arguments(["-f", "x.wav", "--no-diarize"]).diarize is False


def _skipping_outcome(reason: SkipReason = "vad_no_speech") -> TranscriptionOutcome:
    """Build a skipped transcription outcome for CLI stubs.

    Args:
        reason (SkipReason): The typed skip reason to report.

    Returns:
        TranscriptionOutcome: An outcome with an empty transcript.
    """
    empty = pd.DataFrame(columns=pd.Index(["start", "end", "text"]))
    return TranscriptionOutcome(transcript=empty, src_lang="en", skip_reason=reason)


def test_run_main_returns_nonzero_exit_code_when_nothing_was_transcribed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A speech-free file must be distinguishable from a successful run.

    Batch scripts drive the CLI; exiting 0 makes "nothing was transcribed"
    look like "transcribed fine". The code is 3, not 2 — argparse already
    exits 2 for a usage error, and a caller must not confuse a typo'd flag
    with a speech-free file.

    Args:
        monkeypatch (pytest.MonkeyPatch): Overrides the processor and pipeline.
        tmp_path (Path): Temporary working directory.
    """
    monkeypatch.setattr(cli, "FileProcessor", lambda *a, **k: _SpyProcessor())
    monkeypatch.setattr(cli, "transcription_pipeline", lambda **kwargs: _skipping_outcome())

    assert cli._run_main(_args(tmp_path / "clip.wav")) == 3


def test_run_main_returns_zero_for_a_normal_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A run that produced a transcript exits successfully.

    Args:
        monkeypatch (pytest.MonkeyPatch): Overrides the processor and pipeline.
        tmp_path (Path): Temporary working directory.
    """
    monkeypatch.setattr(cli, "FileProcessor", lambda *a, **k: _SpyProcessor())
    df = pd.DataFrame({"start": ["0:00:00"], "end": ["0:00:02"], "text": ["Hello."]})
    monkeypatch.setattr(
        cli, "transcription_pipeline", lambda **kwargs: TranscriptionOutcome(transcript=df, src_lang="en")
    )

    assert cli._run_main(_args(tmp_path / "clip.wav")) == 0


def test_run_main_logs_the_typed_skip_reason(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The warning must name which of the three causes fired.

    Args:
        monkeypatch (pytest.MonkeyPatch): Overrides the processor and pipeline.
        tmp_path (Path): Temporary working directory.
    """
    monkeypatch.setattr(cli, "FileProcessor", lambda *a, **k: _SpyProcessor())
    monkeypatch.setattr(cli, "transcription_pipeline", lambda **kwargs: _skipping_outcome("asr_all_segments_filtered"))

    records: list[str] = []
    sink_id = logger.add(lambda message: records.append(str(message)), level="DEBUG")
    try:
        cli._run_main(_args(tmp_path / "clip.wav"))
    finally:
        logger.remove(sink_id)

    assert any("asr_all_segments_filtered" in record for record in records)


def test_run_main_still_reports_docint_export_when_skipped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A requested docint export must be handled, not silently dropped.

    The early return used to skip the export block entirely, so ``--emit-docint-jsonl``
    produced neither a file nor a word about why.

    Args:
        monkeypatch (pytest.MonkeyPatch): Overrides the processor and pipeline.
        tmp_path (Path): Temporary working directory.
    """
    calls: list[Path] = []
    monkeypatch.setattr(cli, "FileProcessor", lambda *a, **k: _SpyProcessor())
    monkeypatch.setattr(cli, "transcription_pipeline", lambda **kwargs: _skipping_outcome())
    monkeypatch.setattr(cli, "_emit_docint_jsonl", lambda **kwargs: calls.append(kwargs["output_path"]))

    cli._run_main(_args(tmp_path / "clip.wav", emit_docint_jsonl=tmp_path / "out.jsonl"))

    assert calls == [tmp_path / "out.jsonl"]
