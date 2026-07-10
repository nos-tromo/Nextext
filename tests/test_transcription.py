"""Tests for the external Whisper transcription module."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

from nextext.core import audio, transcription
from nextext.core.transcription import (
    ExternalWhisperTranscriber,
    _ends_with_punctuation,
    _merge_transcriptions_by_sentence,
    _seconds_to_time,
)


@contextmanager
def _passthrough_normalize(file_path: Path) -> Iterator[Path]:
    """Stand-in for normalize_for_transcription that yields the input unchanged.

    Args:
        file_path (Path): The path passed through verbatim.

    Yields:
        Path: ``file_path`` unchanged.
    """
    yield file_path


# ---------------------------------------------------------------------------
# Sentence merging
# ---------------------------------------------------------------------------


def test_merge_transcriptions_keeps_final_sentence_without_terminal_punctuation() -> None:
    """_merge_transcriptions_by_sentence keeps the final sentence without terminal punctuation."""
    data = pd.DataFrame(
        [
            {"start": "0:00:00", "end": "0:00:01", "text": "hello"},
            {"start": "0:00:01", "end": "0:00:02", "text": "world"},
        ]
    )

    merged = _merge_transcriptions_by_sentence(data)

    assert list(merged.columns) == ["start", "end", "text"]
    assert merged.to_dict("records") == [{"start": "0:00:00", "end": "0:00:02", "text": "hello world"}]


def test_merge_transcriptions_handles_arabic_question_mark() -> None:
    """_merge_transcriptions_by_sentence treats Arabic question marks (؟) as sentence terminators."""
    data = pd.DataFrame(
        [
            {"start": "0:00:00", "end": "0:00:01", "text": "مرحبا"},
            {"start": "0:00:01", "end": "0:00:02", "text": "كيف الحال؟"},
        ]
    )

    merged = _merge_transcriptions_by_sentence(data)

    assert merged.to_dict("records") == [{"start": "0:00:00", "end": "0:00:02", "text": "مرحبا كيف الحال؟"}]


def test_merge_transcriptions_preserves_speaker_column() -> None:
    """Test that sentence merging keeps speaker labels for diarized rows."""
    data = pd.DataFrame(
        [
            {
                "start": "0:00:00",
                "end": "0:00:01",
                "speaker": "SPEAKER_00",
                "text": "hello",
            },
            {
                "start": "0:00:01",
                "end": "0:00:02",
                "speaker": "SPEAKER_00",
                "text": "world.",
            },
        ]
    )

    merged = _merge_transcriptions_by_sentence(data)

    assert list(merged.columns) == ["start", "end", "speaker", "text"]
    assert merged.to_dict("records") == [
        {
            "start": "0:00:00",
            "end": "0:00:02",
            "speaker": "SPEAKER_00",
            "text": "hello world.",
        }
    ]


def test_merge_transcriptions_splits_when_speaker_changes() -> None:
    """Test that sentence merging does not blend text across speakers."""
    data = pd.DataFrame(
        [
            {
                "start": "0:00:00",
                "end": "0:00:01",
                "speaker": "SPEAKER_00",
                "text": "hello",
            },
            {
                "start": "0:00:01",
                "end": "0:00:02",
                "speaker": "SPEAKER_01",
                "text": "world",
            },
        ]
    )

    merged = _merge_transcriptions_by_sentence(data)

    assert merged.to_dict("records") == [
        {
            "start": "0:00:00",
            "end": "0:00:01",
            "speaker": "SPEAKER_00",
            "text": "hello",
        },
        {
            "start": "0:00:01",
            "end": "0:00:02",
            "speaker": "SPEAKER_01",
            "text": "world",
        },
    ]


# ---------------------------------------------------------------------------
# Module-level helper functions
# ---------------------------------------------------------------------------


def test_seconds_to_time_formats_correctly() -> None:
    """Test that _seconds_to_time converts seconds to HH:MM:SS format."""
    assert _seconds_to_time(0) == "0:00:00"
    assert _seconds_to_time(3661) == "1:01:01"
    assert _seconds_to_time(90.4) == "0:01:30"


def test_ends_with_punctuation_detects_terminal_marks() -> None:
    """Test that _ends_with_punctuation handles Latin and Arabic punctuation."""
    assert _ends_with_punctuation("hello.") is True
    assert _ends_with_punctuation("hello!") is True
    assert _ends_with_punctuation("hello?") is True
    assert _ends_with_punctuation("مرحبا؟") is True
    assert _ends_with_punctuation("hello") is False
    assert _ends_with_punctuation("") is False


# ---------------------------------------------------------------------------
# ExternalWhisperTranscriber
# ---------------------------------------------------------------------------


def test_external_transcriber_transcript_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that ExternalWhisperTranscriber.transcript_output builds the correct DataFrame.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture for patching.
    """
    transcriber = ExternalWhisperTranscriber.__new__(ExternalWhisperTranscriber)
    transcriber.file_path = transcription.Path("dummy.wav")
    transcriber.src_lang = "en"
    transcriber.task = "transcribe"
    transcriber._model_id = "whisper-1"
    transcriber.n_speakers = 1
    transcriber.start_column = "start"
    transcriber.end_column = "end"
    transcriber.speaker_column = "speaker"
    transcriber.text_column = "text"
    transcriber.transcription_result = {
        "segments": [
            {"start": 0.0, "end": 1.5, "text": "Hello world."},
            {"start": 1.5, "end": 3.0, "text": "How are you?"},
        ]
    }

    df = transcriber.transcript_output()

    assert list(df.columns) == ["start", "end", "text"]
    assert len(df) == 2
    assert df.iloc[0]["text"] == "Hello world."
    assert df.iloc[1]["text"] == "How are you?"


def test_external_transcriber_transcript_output_includes_speakers() -> None:
    """Diarized segments surface a speaker column in the output DataFrame."""
    transcriber = _make_external_transcriber()
    transcriber.n_speakers = 2
    transcriber.transcription_result = {
        "segments": [
            {"start": 0.0, "end": 1.5, "text": "Hello there.", "speaker": "SPEAKER_00"},
            {"start": 1.5, "end": 3.0, "text": "General Kenobi.", "speaker": "SPEAKER_01"},
        ]
    }

    df = transcriber.transcript_output()

    assert list(df.columns) == ["start", "end", "speaker", "text"]
    assert df["speaker"].tolist() == ["SPEAKER_00", "SPEAKER_01"]


def test_external_transcriber_transcript_output_hides_single_distinct_speaker() -> None:
    """A single distinct speaker label yields a clean, speaker-free transcript.

    Regression guard for the ``n_speakers``-based drop rule this replaces:
    previously a single labeled speaker with ``n_speakers >= 2`` still
    surfaced a (redundant) single-value speaker column. The rule is now
    based purely on the count of *distinct* speaker labels, independent of
    ``n_speakers``, so this must hide the column even though ``n_speakers``
    is 2 here.
    """
    transcriber = _make_external_transcriber()
    transcriber.n_speakers = 2
    transcriber.transcription_result = {
        "segments": [
            {"start": 0.0, "end": 1.5, "text": "Hello there.", "speaker": "SPEAKER_00"},
            {"start": 1.5, "end": 3.0, "text": "Still me.", "speaker": "SPEAKER_00"},
        ]
    }

    df = transcriber.transcript_output()

    assert list(df.columns) == ["start", "end", "text"]
    assert "speaker" not in df.columns


def test_external_transcriber_transcription_populates_src_lang(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that ExternalWhisperTranscriber.transcription() populates src_lang from the response.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture for patching.
    """
    seg = SimpleNamespace(start=0.0, end=1.0, text="Hola.", no_speech_prob=0.1)
    fake_response = SimpleNamespace(segments=[seg], words=[], language="es")

    fake_client = MagicMock()
    fake_client.audio.transcriptions.create.return_value = fake_response

    transcriber = ExternalWhisperTranscriber.__new__(ExternalWhisperTranscriber)
    transcriber.file_path = transcription.Path(__file__)  # use an existing file
    transcriber.src_lang = None
    transcriber.task = "transcribe"
    transcriber._model_id = "whisper-1"
    transcriber._client = None
    transcriber.transcription_result = None

    # Bypass lazy client creation
    monkeypatch.setattr(
        type(transcriber),
        "_get_client",
        property(lambda self: fake_client),
    )
    # The external /vad guard would otherwise screen this test file; stub it
    # to report speech so the test focuses on src_lang propagation only.
    monkeypatch.setattr(transcription, "has_speech", lambda _path: True)
    monkeypatch.setattr(transcription, "normalize_for_transcription", _passthrough_normalize)

    transcriber.transcription()

    assert transcriber.src_lang == "es"
    assert transcriber.transcription_result is not None
    assert len(transcriber.transcription_result["segments"]) == 1


def test_external_transcriber_populates_src_lang_from_empty_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty-string ``src_lang`` is treated as unspecified, capturing the detected language.

    Regression: the API layer passes ``src_lang=""`` (not ``None``) when the
    caller picks no language, but the detected language was only captured when
    ``src_lang is None`` -- so ``resolved_src_lang`` came back empty and the
    frontend showed no language.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture.
    """
    seg = SimpleNamespace(start=0.0, end=1.0, text="Hola.", no_speech_prob=0.1)
    fake_response = SimpleNamespace(segments=[seg], words=[], language="es")

    fake_client = MagicMock()
    fake_client.audio.transcriptions.create.return_value = fake_response

    transcriber = ExternalWhisperTranscriber.__new__(ExternalWhisperTranscriber)
    transcriber.file_path = transcription.Path(__file__)  # use an existing file
    transcriber.src_lang = ""  # API sends "" for "auto-detect", not None
    transcriber.task = "transcribe"
    transcriber._model_id = "whisper-1"
    transcriber._client = None
    transcriber.transcription_result = None

    # Bypass lazy client creation
    monkeypatch.setattr(
        type(transcriber),
        "_get_client",
        property(lambda self: fake_client),
    )
    # The external /vad guard would otherwise screen this test file; stub it
    # to report speech so the test focuses on src_lang propagation only.
    monkeypatch.setattr(transcription, "has_speech", lambda _path: True)
    monkeypatch.setattr(transcription, "normalize_for_transcription", _passthrough_normalize)

    transcriber.transcription()

    assert transcriber.src_lang == "es"


def test_external_transcriber_normalizes_full_language_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI returns full language names; ``src_lang`` must normalize to ISO.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture for
            patching the lazy client and the /vad guard.
    """
    seg = SimpleNamespace(start=0.0, end=1.0, text="Hallo.", no_speech_prob=0.1)
    fake_response = SimpleNamespace(segments=[seg], words=[], language="german")

    fake_client = MagicMock()
    fake_client.audio.transcriptions.create.return_value = fake_response

    transcriber = ExternalWhisperTranscriber.__new__(ExternalWhisperTranscriber)
    transcriber.file_path = transcription.Path(__file__)
    transcriber.src_lang = None
    transcriber.task = "transcribe"
    transcriber._model_id = "whisper-1"
    transcriber._client = None
    transcriber.transcription_result = None

    monkeypatch.setattr(
        type(transcriber),
        "_get_client",
        property(lambda self: fake_client),
    )
    monkeypatch.setattr(transcription, "has_speech", lambda _path: True)
    monkeypatch.setattr(transcription, "normalize_for_transcription", _passthrough_normalize)

    transcriber.transcription()

    assert transcriber.src_lang == "de"


# ---------------------------------------------------------------------------
# Shared VAD / no_speech_prob helpers
# ---------------------------------------------------------------------------


def _make_external_transcriber() -> ExternalWhisperTranscriber:
    """Build a minimally-initialised ExternalWhisperTranscriber for tests.

    Uses ``__new__`` to bypass the real ``__init__`` (which would read env
    vars and construct an OpenAI client). All attributes read by
    :meth:`ExternalWhisperTranscriber.transcription` are explicitly set.

    Returns:
        ExternalWhisperTranscriber: An instance whose ``file_path`` points
        at this test module. Callers always stub ``has_speech`` before
        ``transcription()`` so the file is never actually screened.
    """
    transcriber = ExternalWhisperTranscriber.__new__(ExternalWhisperTranscriber)
    transcriber.file_path = transcription.Path(__file__)
    transcriber.src_lang = None
    transcriber.task = "transcribe"
    transcriber._model_id = "whisper-1"
    transcriber.n_speakers = 1
    transcriber._client = None
    transcriber.start_column = "start"
    transcriber.end_column = "end"
    transcriber.speaker_column = "speaker"
    transcriber.text_column = "text"
    transcriber.transcription_result = None
    return transcriber


def test_filter_no_speech_segments_drops_high_prob_entries() -> None:
    """``_filter_no_speech_segments`` must honour the 0.6 threshold exactly.

    Walks the boundary of :data:`NO_SPEECH_THRESHOLD` (``<=`` is the
    keep-rule) and verifies that missing ``no_speech_prob`` defaults to
    zero (kept). The assertion compares surviving segment texts in order,
    so it catches both incorrect drops and reordering.
    """
    segments = [
        {"start": 0.0, "end": 1.0, "text": "keep low", "no_speech_prob": 0.1},
        {"start": 1.0, "end": 2.0, "text": "drop high", "no_speech_prob": 0.9},
        {"start": 2.0, "end": 3.0, "text": "keep missing"},
        {"start": 3.0, "end": 4.0, "text": "drop boundary", "no_speech_prob": 0.61},
        {"start": 4.0, "end": 5.0, "text": "keep boundary", "no_speech_prob": 0.6},
    ]

    filtered = transcription._filter_no_speech_segments(segments)

    assert [seg["text"] for seg in filtered] == [
        "keep low",
        "keep missing",
        "keep boundary",
    ]


def test_filter_no_speech_segments_returns_input_when_nothing_dropped() -> None:
    """Identity optimization must hold when every segment passes the filter.

    The helper is invoked on the happy path of every transcription; the
    contract is that it returns the *same* list object (not a copy) when
    nothing needs dropping, to avoid allocation per file. The ``is`` check
    pins that contract down so a well-meaning refactor cannot silently
    regress it.
    """
    segments = [
        {"start": 0.0, "end": 1.0, "text": "a", "no_speech_prob": 0.1},
        {"start": 1.0, "end": 2.0, "text": "b"},
    ]

    assert transcription._filter_no_speech_segments(segments) is segments


# ---------------------------------------------------------------------------
# ExternalWhisperTranscriber VAD guard + no_speech_prob post-filter
# ---------------------------------------------------------------------------


def test_external_transcriber_skips_when_vad_reports_no_speech(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Speech-free audio must short-circuit the external request entirely.

    The whole point of guarding the external path is to avoid paying for
    a transcription request whose result would be hallucinated text. The
    test stubs ``has_speech`` to report no speech, runs
    ``transcription()``, and asserts that:

    * the method returns cleanly with ``transcription_result`` holding an
      empty segment list;
    * the OpenAI client was never dereferenced — verified by patching
      ``_get_client`` with a property that raises if accessed.

    Args:
        monkeypatch (pytest.MonkeyPatch): Overrides ``has_speech`` and the client property.
    """
    transcriber = _make_external_transcriber()
    monkeypatch.setattr(transcription, "has_speech", lambda _path: False)

    def _client_must_not_be_built(_self: Any) -> Any:
        raise AssertionError("external API must not be contacted for silent audio")

    monkeypatch.setattr(
        type(transcriber),
        "_get_client",
        property(_client_must_not_be_built),
    )

    transcriber.transcription()

    assert transcriber.transcription_result == {"segments": [], "words": []}


def test_external_transcriber_applies_no_speech_prob_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The external response segments must pass through the no_speech_prob filter.

    Before the refactor, the external path stripped ``no_speech_prob``
    from the response at segment-build time, which made post-filtering
    impossible. This test feeds a fake response with one low-prob
    segment and one high-prob segment, then asserts:

    * only the low-prob segment survives in ``transcription_result``;
    * ``no_speech_prob`` is preserved on the survivor so downstream
      consumers can inspect it if they wish.

    The /vad guard is bypassed with a ``True`` stub so the test
    focuses exclusively on the post-filter behaviour.

    Args:
        monkeypatch (pytest.MonkeyPatch): Overrides ``has_speech`` and the client property.
    """
    kept = SimpleNamespace(start=0.0, end=1.0, text="Hello", no_speech_prob=0.1)
    dropped = SimpleNamespace(start=1.0, end=2.0, text="hallucinated.", no_speech_prob=0.95)
    fake_response = SimpleNamespace(segments=[kept, dropped], words=[], language="en")

    fake_client = MagicMock()
    fake_client.audio.transcriptions.create.return_value = fake_response

    transcriber = _make_external_transcriber()
    monkeypatch.setattr(transcription, "has_speech", lambda _path: True)
    monkeypatch.setattr(transcription, "normalize_for_transcription", _passthrough_normalize)
    monkeypatch.setattr(
        type(transcriber),
        "_get_client",
        property(lambda self: fake_client),
    )

    transcriber.transcription()

    assert transcriber.transcription_result is not None
    survivors = transcriber.transcription_result["segments"]
    assert len(survivors) == 1
    assert survivors[0]["text"] == "Hello"
    assert survivors[0]["no_speech_prob"] == pytest.approx(0.1)


def test_external_transcriber_tolerates_missing_no_speech_prob(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider responses without ``no_speech_prob`` must default to zero.

    Not every OpenAI-compatible server returns ``no_speech_prob`` on each
    segment. The external path uses ``getattr(..., 0.0)`` so missing
    attributes are treated as ``0.0`` (clean — kept by the filter). This
    test verifies that contract end-to-end: a response segment with no
    ``no_speech_prob`` attribute must land in the result and its stored
    probability must be exactly ``0.0``.

    Args:
        monkeypatch (pytest.MonkeyPatch): Overrides ``has_speech`` and the client property.
    """
    seg = SimpleNamespace(start=0.0, end=1.0, text="Hola.")
    fake_response = SimpleNamespace(segments=[seg], words=[], language="es")

    fake_client = MagicMock()
    fake_client.audio.transcriptions.create.return_value = fake_response

    transcriber = _make_external_transcriber()
    monkeypatch.setattr(transcription, "has_speech", lambda _path: True)
    monkeypatch.setattr(transcription, "normalize_for_transcription", _passthrough_normalize)
    monkeypatch.setattr(
        type(transcriber),
        "_get_client",
        property(lambda self: fake_client),
    )

    transcriber.transcription()

    assert transcriber.transcription_result is not None
    segments = transcriber.transcription_result["segments"]
    assert len(segments) == 1
    assert segments[0]["no_speech_prob"] == 0.0


def test_transcription_normalizes_before_upload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """transcription() routes the upload through normalize_for_transcription.

    Args:
        monkeypatch (pytest.MonkeyPatch): Patches the client, VAD guard, and
            the normalization seam.
    """
    seg = SimpleNamespace(start=0.0, end=1.0, text="Hi.", no_speech_prob=0.1)
    fake_response = SimpleNamespace(segments=[seg], words=[], language="en")
    fake_client = MagicMock()
    fake_client.audio.transcriptions.create.return_value = fake_response

    transcriber = ExternalWhisperTranscriber.__new__(ExternalWhisperTranscriber)
    transcriber.file_path = transcription.Path(__file__)
    transcriber.src_lang = "en"
    transcriber.task = "transcribe"
    transcriber._model_id = "whisper-1"
    transcriber._client = None
    transcriber.transcription_result = None

    monkeypatch.setattr(type(transcriber), "_get_client", property(lambda self: fake_client))
    monkeypatch.setattr(transcription, "has_speech", lambda _path: True)

    seen: list[Path] = []

    @contextmanager
    def _spy(file_path: Path) -> Iterator[Path]:
        seen.append(file_path)
        yield file_path

    monkeypatch.setattr(transcription, "normalize_for_transcription", _spy)

    transcriber.transcription()

    assert seen == [transcriber.file_path]
    fake_client.audio.transcriptions.create.assert_called_once()


def test_transcription_requests_word_granularity_and_captures_words(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The transcribe call asks for segment+word granularity and stores words.

    Args:
        monkeypatch (pytest.MonkeyPatch): Patches the client, VAD guard, and
            the normalization seam.
    """
    seg = SimpleNamespace(start=0.0, end=1.0, text="hi", no_speech_prob=0.0)
    word = SimpleNamespace(word="hi", start=0.0, end=0.4)
    fake_response = SimpleNamespace(segments=[seg], words=[word], language="en")
    fake_client = MagicMock()
    fake_client.audio.transcriptions.create.return_value = fake_response

    transcriber = ExternalWhisperTranscriber.__new__(ExternalWhisperTranscriber)
    transcriber.file_path = transcription.Path(__file__)
    transcriber.src_lang = "en"
    transcriber.task = "transcribe"
    transcriber._model_id = "whisper-1"
    transcriber._client = None
    transcriber.transcription_result = None

    monkeypatch.setattr(type(transcriber), "_get_client", property(lambda self: fake_client))
    monkeypatch.setattr(transcription, "has_speech", lambda _path: True)
    monkeypatch.setattr(transcription, "normalize_for_transcription", _passthrough_normalize)

    transcriber.transcription()

    _, kwargs = fake_client.audio.transcriptions.create.call_args
    assert kwargs["timestamp_granularities"] == ["segment", "word"]
    assert transcriber.transcription_result is not None
    assert transcriber.transcription_result["words"] == [{"word": "hi", "start": 0.0, "end": 0.4}]


def test_transcription_defaults_words_to_empty_when_response_omits_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A response object without a `words` attribute yields an empty words list.

    Args:
        monkeypatch (pytest.MonkeyPatch): Patches the client, VAD guard, and
            the normalization seam.
    """
    seg = SimpleNamespace(start=0.0, end=1.0, text="hi", no_speech_prob=0.0)
    fake_response = SimpleNamespace(segments=[seg], language="en")  # no `words` attribute
    fake_client = MagicMock()
    fake_client.audio.transcriptions.create.return_value = fake_response

    transcriber = ExternalWhisperTranscriber.__new__(ExternalWhisperTranscriber)
    transcriber.file_path = transcription.Path(__file__)
    transcriber.src_lang = "en"
    transcriber.task = "transcribe"
    transcriber._model_id = "whisper-1"
    transcriber._client = None
    transcriber.transcription_result = None

    monkeypatch.setattr(type(transcriber), "_get_client", property(lambda self: fake_client))
    monkeypatch.setattr(transcription, "has_speech", lambda _path: True)
    monkeypatch.setattr(transcription, "normalize_for_transcription", _passthrough_normalize)

    transcriber.transcription()

    assert transcriber.transcription_result["words"] == []
    assert transcriber.transcription_result["segments"][0]["text"] == "hi"


def test_transcription_raises_on_undecodable_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An AudioDecodeError from normalization aborts before any API call.

    Args:
        monkeypatch (pytest.MonkeyPatch): Patches the client, VAD guard, and
            the normalization seam to raise.
    """
    fake_client = MagicMock()
    transcriber = ExternalWhisperTranscriber.__new__(ExternalWhisperTranscriber)
    transcriber.file_path = transcription.Path("voice.ogg")
    transcriber.src_lang = None
    transcriber.task = "transcribe"
    transcriber._model_id = "whisper-1"
    transcriber._client = None
    transcriber.transcription_result = None

    monkeypatch.setattr(type(transcriber), "_get_client", property(lambda self: fake_client))
    monkeypatch.setattr(transcription, "has_speech", lambda _path: True)

    @contextmanager
    def _raise(file_path: Path) -> Iterator[Path]:
        raise audio.AudioDecodeError(f"Could not decode '{file_path.name}'.")
        yield file_path  # unreachable; only here so @contextmanager treats this as a generator

    monkeypatch.setattr(transcription, "normalize_for_transcription", _raise)

    with pytest.raises(audio.AudioDecodeError):
        transcriber.transcription()

    fake_client.audio.transcriptions.create.assert_not_called()
