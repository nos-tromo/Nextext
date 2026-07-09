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

from nextext import cli


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
        "speakers": 1,
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
    monkeypatch.setattr(cli, "transcription_pipeline", lambda **kwargs: (df, "en"))

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
    monkeypatch.setattr(cli, "transcription_pipeline", lambda **kwargs: (empty, None))

    cli._run_main(_args(tmp_path / "clip.wav"))

    (processor,) = created
    assert len(processor.transcript_writes) == 1
    assert "transcript" not in processor.file_output_labels
