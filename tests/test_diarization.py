"""Tests for the speaker-diarization agent (HTTP client + overlap alignment)."""

from pathlib import Path
from typing import Any

import httpx
import pytest

from nextext.core import diarization
from nextext.core.diarization import (
    assign_speakers_by_overlap,
    build_speaker_segments,
    canonicalize_speaker_labels,
    diarize_file,
    gate_turns_by_vad,
)

# ---------------------------------------------------------------------------
# assign_speakers_by_overlap
# ---------------------------------------------------------------------------


def test_assign_speakers_uses_maximum_overlap() -> None:
    """The speaker with the greatest cumulative overlap is assigned to a segment."""
    segments: list[dict[str, Any]] = [{"start": 0.0, "end": 2.0, "text": "hello world"}]
    turns = [
        {"start": 0.0, "end": 1.8, "speaker": "SPEAKER_00"},  # 1.8s overlap
        {"start": 1.8, "end": 2.0, "speaker": "SPEAKER_01"},  # 0.2s overlap
    ]

    assign_speakers_by_overlap(segments, turns)

    assert segments[0]["speaker"] == "SPEAKER_00"


def test_assign_speakers_leaves_non_overlapping_segment_untouched() -> None:
    """A segment that overlaps no diarization turn gains no speaker key."""
    segments: list[dict[str, Any]] = [{"start": 5.0, "end": 6.0, "text": "x"}]
    turns = [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]

    assign_speakers_by_overlap(segments, turns)

    assert "speaker" not in segments[0]


def test_assign_speakers_empty_segments_is_noop() -> None:
    """Aligning against zero transcription segments is a safe no-op."""
    segments: list[dict[str, Any]] = []

    assign_speakers_by_overlap(segments, [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}])

    assert segments == []


# ---------------------------------------------------------------------------
# diarize_file
# ---------------------------------------------------------------------------


def test_diarize_file_returns_empty_when_base_unset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """With no diarization endpoint configured (dedicated or central), it issues no request.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching env vars and httpx.
        tmp_path (Path): Temporary directory fixture for the audio file.
    """
    monkeypatch.delenv("DIARIZE_API_BASE", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)

    def fail_post(url: str, **kwargs: Any) -> httpx.Response:
        raise AssertionError("httpx.post must not be called when no diarization endpoint is configured")

    monkeypatch.setattr(diarization.httpx, "post", fail_post)
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"data")

    assert diarize_file(audio, max_speakers=3) == []


def test_diarize_file_posts_correctly_and_parses(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The request targets /diarize with the speaker field, bearer auth, and timeout.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching env vars and httpx.
        tmp_path (Path): Temporary directory fixture for the audio file.
    """
    monkeypatch.setenv("DIARIZE_API_BASE", "http://router:9000/")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    monkeypatch.delenv("DIARIZE_TIMEOUT", raising=False)
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        captured["url"] = url
        captured.update(kwargs)
        return httpx.Response(
            200,
            json={
                "segments": [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}],
                "speakers": ["SPEAKER_00"],
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(diarization.httpx, "post", fake_post)
    audio = tmp_path / "clip.mp4"
    audio.write_bytes(b"bytes")

    segments = diarize_file(audio, max_speakers=4)

    assert segments == [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]
    assert captured["url"] == "http://router:9000/diarize"
    assert captured["data"] == {"max_speakers": 4}
    assert captured["headers"]["Authorization"] == "Bearer sk-secret"
    assert captured["timeout"] == 600.0
    assert captured["files"]["file"][0] == "clip.mp4"


def test_diarize_file_omits_authorization_without_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """No Authorization header is sent when OPENAI_API_KEY is empty.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching env vars and httpx.
        tmp_path (Path): Temporary directory fixture for the audio file.
    """
    monkeypatch.setenv("DIARIZE_API_BASE", "http://router:9000")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        captured.update(kwargs)
        return httpx.Response(200, json={"segments": []}, request=httpx.Request("POST", url))

    monkeypatch.setattr(diarization.httpx, "post", fake_post)
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"x")

    diarize_file(audio, max_speakers=2)

    assert "Authorization" not in captured["headers"]


def test_diarize_file_swallows_http_status_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A non-2xx response is logged and yields an empty list.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching env vars and httpx.
        tmp_path (Path): Temporary directory fixture for the audio file.
    """
    monkeypatch.setenv("DIARIZE_API_BASE", "http://router:9000")

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(500, text="boom", request=httpx.Request("POST", url))

    monkeypatch.setattr(diarization.httpx, "post", fake_post)
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"x")

    assert diarize_file(audio, max_speakers=2) == []


def test_diarize_file_swallows_transport_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A transport error (e.g. connection refused) is logged and yields an empty list.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching env vars and httpx.
        tmp_path (Path): Temporary directory fixture for the audio file.
    """
    monkeypatch.setenv("DIARIZE_API_BASE", "http://router:9000")

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    monkeypatch.setattr(diarization.httpx, "post", fake_post)
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"x")

    assert diarize_file(audio, max_speakers=2) == []


def test_diarize_file_handles_non_dict_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A JSON payload that is not an object is rejected and yields an empty list.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching env vars and httpx.
        tmp_path (Path): Temporary directory fixture for the audio file.
    """
    monkeypatch.setenv("DIARIZE_API_BASE", "http://router:9000")

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(200, json=["not", "a", "dict"], request=httpx.Request("POST", url))

    monkeypatch.setattr(diarization.httpx, "post", fake_post)
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"x")

    assert diarize_file(audio, max_speakers=2) == []


# ---------------------------------------------------------------------------
# canonicalize_speaker_labels
# ---------------------------------------------------------------------------


def test_canonicalize_numbers_by_first_appearance() -> None:
    """Raw pyannote labels renumber to contiguous Speaker N by earliest start."""
    turns = [
        {"start": 5.0, "end": 6.0, "speaker": "SPEAKER_02"},
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
        {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_02"},
    ]

    result = canonicalize_speaker_labels(turns)

    # First voice heard (start=0.0, SPEAKER_00) -> Speaker 1; next new voice -> Speaker 2.
    assert [t["speaker"] for t in result] == ["Speaker 2", "Speaker 1", "Speaker 2"]
    # Original order and timings are preserved; only the label string changes.
    assert [t["start"] for t in result] == [5.0, 0.0, 1.0]


def test_canonicalize_empty_is_empty() -> None:
    """No turns canonicalizes to no turns."""
    assert canonicalize_speaker_labels([]) == []


# ---------------------------------------------------------------------------
# build_speaker_segments
# ---------------------------------------------------------------------------

_TURNS = [
    {"start": 0.0, "end": 1.0, "speaker": "Speaker 1"},
    {"start": 1.0, "end": 2.0, "speaker": "Speaker 2"},
]


def test_build_keeps_single_speaker_segment_verbatim() -> None:
    """A segment whose words share one speaker is emitted with exact text preserved."""
    segments = [{"start": 0.0, "end": 0.9, "text": "hello world"}]
    words = [
        {"word": "hello", "start": 0.0, "end": 0.4},
        {"word": "world", "start": 0.4, "end": 0.8},
    ]

    result = build_speaker_segments(segments, words, _TURNS)

    assert result == [{"start": 0.0, "end": 0.9, "text": "hello world", "speaker": "Speaker 1"}]


def test_build_splits_mixed_speaker_segment_at_word() -> None:
    """A segment spanning a speaker change splits at the exact word."""
    segments = [{"start": 0.0, "end": 2.0, "text": "hi there"}]
    words = [
        {"word": "hi", "start": 0.0, "end": 0.4},  # midpoint 0.2 -> Speaker 1
        {"word": "there", "start": 1.2, "end": 1.8},  # midpoint 1.5 -> Speaker 2
    ]

    result = build_speaker_segments(segments, words, _TURNS)

    assert result == [
        {"start": 0.0, "end": 0.4, "text": "hi", "speaker": "Speaker 1"},
        {"start": 1.2, "end": 1.8, "text": "there", "speaker": "Speaker 2"},
    ]


def test_build_falls_back_to_segment_level_without_words() -> None:
    """With no word timestamps, assignment is segment-level max overlap."""
    segments = [{"start": 0.0, "end": 0.9, "text": "hello"}]

    result = build_speaker_segments(segments, [], _TURNS)

    assert result == [{"start": 0.0, "end": 0.9, "text": "hello", "speaker": "Speaker 1"}]
    # Input is not mutated (a copy is returned).
    assert "speaker" not in segments[0]


def test_build_unlabeled_word_inherits_neighbouring_run() -> None:
    """A word overlapping no turn does not force a split; it joins the current run."""
    segments = [{"start": 0.0, "end": 3.0, "text": "a b c"}]
    words = [
        {"word": "a", "start": 0.0, "end": 0.4},  # Speaker 1
        {"word": "b", "start": 2.2, "end": 2.4},  # overlaps no turn -> None
        {"word": "c", "start": 2.5, "end": 2.8},  # overlaps no turn -> None
    ]
    turns = [{"start": 0.0, "end": 1.0, "speaker": "Speaker 1"}]

    result = build_speaker_segments(segments, words, turns)

    # Single distinct speaker (Speaker 1) -> verbatim segment, exact text.
    assert result == [{"start": 0.0, "end": 3.0, "text": "a b c", "speaker": "Speaker 1"}]


def test_build_none_word_between_two_speakers_folds_into_run() -> None:
    """A no-overlap word between two different speakers folds into a run, not a split."""
    segments = [{"start": 0.0, "end": 3.0, "text": "a b c"}]
    words = [
        {"word": "a", "start": 0.2, "end": 0.4},  # Speaker 1
        {"word": "b", "start": 1.4, "end": 1.6},  # in the turn gap -> None
        {"word": "c", "start": 2.4, "end": 2.6},  # Speaker 2
    ]
    turns = [
        {"start": 0.0, "end": 1.0, "speaker": "Speaker 1"},
        {"start": 2.0, "end": 3.0, "speaker": "Speaker 2"},
    ]

    result = build_speaker_segments(segments, words, turns)

    assert result == [
        {"start": 0.2, "end": 1.6, "text": "a b", "speaker": "Speaker 1"},
        {"start": 2.4, "end": 2.6, "text": "c", "speaker": "Speaker 2"},
    ]


# ---------------------------------------------------------------------------
# gate_turns_by_vad
# ---------------------------------------------------------------------------


def test_gate_turns_splits_turn_straddling_a_gap() -> None:
    """A turn spanning a non-speech gap is cropped to its speech-only pieces."""
    turns = [{"start": 0.0, "end": 10.0, "speaker": "SPEAKER_00"}]
    vad = [(0.0, 3.0), (6.0, 10.0)]  # 3-6 is music / non-speech
    assert gate_turns_by_vad(turns, vad) == [
        {"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"},
        {"start": 6.0, "end": 10.0, "speaker": "SPEAKER_00"},
    ]


def test_gate_turns_drops_turn_entirely_in_non_speech() -> None:
    """A turn overlapping no speech interval is dropped."""
    turns = [{"start": 4.0, "end": 5.0, "speaker": "SPEAKER_01"}]
    assert gate_turns_by_vad(turns, [(0.0, 3.0), (6.0, 10.0)]) == []


def test_gate_turns_keeps_turn_fully_in_speech_unchanged() -> None:
    """A turn wholly within a speech interval is returned unchanged, speaker preserved."""
    turns = [{"start": 1.0, "end": 2.0, "speaker": "SPEAKER_00"}]
    assert gate_turns_by_vad(turns, [(0.0, 3.0)]) == [{"start": 1.0, "end": 2.0, "speaker": "SPEAKER_00"}]


def test_gate_turns_empty_intervals_is_failsafe_passthrough() -> None:
    """Empty VAD intervals never blank the turns — they pass through unchanged."""
    turns = [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}]
    assert gate_turns_by_vad(turns, []) == turns


def test_gate_turns_splits_turn_across_multiple_gaps() -> None:
    """A turn spanning several non-speech gaps yields one fragment per speech interval."""
    turns = [{"start": 0.0, "end": 20.0, "speaker": "SPEAKER_00"}]
    vad = [(0.0, 3.0), (6.0, 9.0), (15.0, 18.0)]
    assert gate_turns_by_vad(turns, vad) == [
        {"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"},
        {"start": 6.0, "end": 9.0, "speaker": "SPEAKER_00"},
        {"start": 15.0, "end": 18.0, "speaker": "SPEAKER_00"},
    ]
