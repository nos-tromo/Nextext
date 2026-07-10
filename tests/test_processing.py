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
    rule = "=" * 40
    assert txt.read_text(encoding="utf-8") == f"{rule}\n[00:00:00 - 00:00:02]  S1\n{rule}\nHello world.\n"
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
    rule = "=" * 40
    assert transcript_txt == f"{rule}\n[00:00:00 - 00:00:02]  S1\n{rule}\nHello world.\n"
    assert "Hallo Welt." not in transcript_txt
    assert translation_txt == f"{rule}\n[00:00:00 - 00:00:02]  S1\n{rule}\nHallo Welt.\n"
    assert "Hallo Welt." in translation_txt
    # The combined CSV still carries both columns side by side.
    combined = pd.read_csv(out / "clip_transcript.csv")
    assert list(combined.columns) == ["start", "end", "speaker", "text", "translation"]


def test_write_transcript_output_dotted_stem_no_collision(tmp_path: Path) -> None:
    """A dotted input stem must not collapse the two TXT files onto one path.

    ``Path.with_suffix`` treats everything after the first dot as the suffix, so
    ``clip.v2_transcript`` and ``clip.v2_translation`` would both become
    ``clip.txt`` and silently overwrite each other. The f-string path keeps the
    full stem and label distinct.
    """
    processor = FileProcessor(file_path=Path("clip.v2.wav"), output_dir=tmp_path)
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
    out = tmp_path / "clip.v2"
    transcript_txt = out / "clip.v2_transcript.txt"
    translation_txt = out / "clip.v2_translation.txt"
    assert transcript_txt.exists()
    assert translation_txt.exists()
    assert transcript_txt != translation_txt
    assert "Hallo Welt." not in transcript_txt.read_text(encoding="utf-8")
    assert "Hallo Welt." in translation_txt.read_text(encoding="utf-8")


def test_write_file_output_dotted_stem_keeps_full_name(tmp_path: Path) -> None:
    """A dotted input stem must not be truncated by ``with_suffix`` for any output.

    ``Path.with_suffix`` treats everything after the first dot as the suffix, so a
    stem like ``episode.2024`` would collapse ``episode.2024_words`` to
    ``episode.csv``. f-string concatenation preserves the full stem + label across
    the csv/xlsx (DataFrame) and txt (string) outputs alike.
    """
    processor = FileProcessor(file_path=Path("episode.2024.wav"), output_dir=tmp_path)
    out = tmp_path / "episode.2024"

    processor.write_file_output(pd.DataFrame({"word": ["hi"], "count": [1]}), "words")
    assert (out / "episode.2024_words.csv").exists()
    assert (out / "episode.2024_words.xlsx").exists()
    assert not (out / "episode.csv").exists()

    processor.write_file_output("A short summary.", "summary")
    assert (out / "episode.2024_summary.txt").exists()
    assert not (out / "episode.txt").exists()


def test_write_transcript_output_empty_transcript_writes_empty_txt(tmp_path: Path) -> None:
    """A no-speech (empty) transcript writes an empty transcript.txt (no segments) and no translation.txt."""
    processor = FileProcessor(file_path=Path("clip.wav"), output_dir=tmp_path)
    df = pd.DataFrame({"start": [], "end": [], "text": []})
    processor.write_transcript_output(df)
    out = tmp_path / "clip"
    txt = out / "clip_transcript.txt"
    assert txt.exists()
    assert txt.read_text(encoding="utf-8") == ""
    assert not (out / "clip_translation.txt").exists()
