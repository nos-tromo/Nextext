"""Tests for the CLI-side FileProcessor transcript output."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nextext.core.processing import FileProcessor


def test_write_transcript_output_transcribe_writes_single_txt(tmp_path: Path) -> None:
    """A transcribe transcript writes combined csv/xlsx plus one transcript.txt."""
    processor = FileProcessor(file_path=Path("clip.wav"), output_dir=tmp_path)
    df = pd.DataFrame(
        {
            "start": ["00:00:00"],
            "end": ["00:00:02"],
            "speaker": ["S1"],
            "text": ["Hello world."],
        }
    )
    processor.write_transcript_output(df)
    out = tmp_path / "clip"
    assert (out / "clip_transcript.csv").exists()
    assert (out / "clip_transcript.xlsx").exists()
    txt = out / "clip_transcript.txt"
    assert txt.exists()
    assert txt.read_text(encoding="utf-8").splitlines()[0] == "start\tend\tspeaker\ttext"
    assert not (out / "clip_translation.txt").exists()


def test_write_transcript_output_translate_writes_two_txt(tmp_path: Path) -> None:
    """A translated transcript writes separate transcript.txt and translation.txt."""
    processor = FileProcessor(file_path=Path("clip.wav"), output_dir=tmp_path)
    df = pd.DataFrame(
        {
            "start": ["00:00:00"],
            "end": ["00:00:02"],
            "speaker": ["S1"],
            "text": ["Hello world."],
            "translation": ["Hallo Welt."],
        }
    )
    processor.write_transcript_output(df)
    out = tmp_path / "clip"
    transcript_txt = (out / "clip_transcript.txt").read_text(encoding="utf-8")
    translation_txt = (out / "clip_translation.txt").read_text(encoding="utf-8")
    assert transcript_txt.splitlines()[0] == "start\tend\tspeaker\ttext"
    assert "Hallo Welt." not in transcript_txt
    assert translation_txt.splitlines()[0] == "start\tend\tspeaker\ttranslation"
    assert "Hallo Welt." in translation_txt
    # The combined CSV still carries both columns side by side.
    combined = pd.read_csv(out / "clip_transcript.csv")
    assert list(combined.columns) == ["start", "end", "speaker", "text", "translation"]
