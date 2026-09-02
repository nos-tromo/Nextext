"""Tests for the CLI-side FileProcessor transcript output."""

from __future__ import annotations

import json
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


def test_write_keyframes_lays_frames_out_like_the_archive(tmp_path: Path) -> None:
    """Frames land in ``{stem}_keyframes/frame_NNN.jpg``, as in ``keyframes.zip``.

    A CLI run and a downloaded archive should be laid out the same way, so a
    reader who has seen one recognizes the other.

    Args:
        tmp_path (Path): Temporary directory fixture.
    """
    processor = FileProcessor(file_path=Path("clip.mp4"), output_dir=tmp_path)

    written = processor.write_keyframes([b"\xff\xd8a", b"\xff\xd8b"])

    assert written is not None
    assert written == processor.output_path / "clip_keyframes"
    assert sorted(p.name for p in written.iterdir()) == ["frame_000.jpg", "frame_001.jpg"]
    assert (written / "frame_001.jpg").read_bytes() == b"\xff\xd8b"


def test_write_keyframes_writes_a_manifest_of_frame_times(tmp_path: Path) -> None:
    """The frames directory names each frame's sampling time, as the zip does.

    Args:
        tmp_path (Path): Temporary directory fixture.
    """
    processor = FileProcessor(file_path=Path("clip.mp4"), output_dir=tmp_path)

    written = processor.write_keyframes([b"\xff\xd8a", b"\xff\xd8b"], times=[0.0, 4.5])

    assert written is not None
    manifest = json.loads((written / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["frames"] == [
        {"file": "frame_000.jpg", "index": 0, "time_sec": 0.0},
        {"file": "frame_001.jpg", "index": 1, "time_sec": 4.5},
    ]


def test_write_keyframes_without_times_writes_no_manifest(tmp_path: Path) -> None:
    """Frames sampled without times get no manifest rather than an empty one.

    Args:
        tmp_path (Path): Temporary directory fixture.
    """
    processor = FileProcessor(file_path=Path("clip.mp4"), output_dir=tmp_path)

    written = processor.write_keyframes([b"\xff\xd8a"])

    assert written is not None
    assert not (written / "manifest.json").exists()


def test_write_keyframes_writes_nothing_for_an_audio_file(tmp_path: Path) -> None:
    """No frames means no directory — an empty folder would read as a failure.

    Args:
        tmp_path (Path): Temporary directory fixture.
    """
    processor = FileProcessor(file_path=Path("clip.wav"), output_dir=tmp_path)

    assert processor.write_keyframes([]) is None
    assert not (processor.output_path / "clip_keyframes").exists()
