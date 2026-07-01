"""Tests for keyframe extraction and even subsampling."""

from pathlib import Path

from nextext.core.keyframes import extract_keyframes, subsample


def test_subsample_evenly_picks_target() -> None:
    """Even subsampling picks target items spread across the input."""
    assert subsample([1, 2, 3, 4, 5, 6, 7, 8], 4) == [1, 3, 5, 7]


def test_subsample_returns_all_when_fewer_than_target() -> None:
    """Fewer items than the target returns them all unchanged."""
    assert subsample([1, 2], 5) == [1, 2]


def test_subsample_zero_target_is_empty() -> None:
    """A non-positive target yields an empty selection."""
    assert subsample([1, 2, 3], 0) == []


def test_extract_keyframes_failsoft_on_non_video(tmp_path: Path) -> None:
    """A non-video or undecodable file yields [] rather than raising."""
    bogus = tmp_path / "not_a_video.bin"
    bogus.write_bytes(b"\x00\x01\x02not media")
    # No video stream / undecodable -> empty list, never raises.
    assert extract_keyframes(bogus, per_minute=4, max_frames=5) == []
