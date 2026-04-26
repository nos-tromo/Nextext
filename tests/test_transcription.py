"""Tests for the openai-whisper transcription module."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
import torch

from nextext.core import transcription
from nextext.core.transcription import (
    ExternalWhisperTranscriber,
    WhisperTranscriber,
    _configure_torch_safe_globals,
    _ends_with_punctuation,
    _merge_transcriptions_by_sentence,
    _seconds_to_time,
)
from nextext.utils.model_registry import REGISTRY


def _build_transcriber() -> WhisperTranscriber:
    """Build a WhisperTranscriber instance for testing.

    Returns:
        WhisperTranscriber: A WhisperTranscriber instance with default column settings.
    """
    transcriber = WhisperTranscriber.__new__(WhisperTranscriber)
    transcriber.start_column = "start"
    transcriber.end_column = "end"
    transcriber.text_column = "text"
    transcriber.speaker_column = "speaker"
    return transcriber


# ---------------------------------------------------------------------------
# Sentence merging (unchanged logic, now delegates to module-level function)
# ---------------------------------------------------------------------------


def test_merge_transcriptions_keeps_final_sentence_without_terminal_punctuation() -> (
    None
):
    """Test that `_merge_transcriptions_by_sentence` correctly merges transcriptions into sentences
    even when the final sentence does not end with terminal punctuation.
    """
    transcriber = _build_transcriber()
    data = pd.DataFrame(
        [
            {"start": "0:00:00", "end": "0:00:01", "text": "hello"},
            {"start": "0:00:01", "end": "0:00:02", "text": "world"},
        ]
    )

    merged = transcriber._merge_transcriptions_by_sentence(data)

    assert list(merged.columns) == ["start", "end", "text"]
    assert merged.to_dict("records") == [
        {"start": "0:00:00", "end": "0:00:02", "text": "hello world"}
    ]


def test_merge_transcriptions_handles_arabic_question_mark() -> None:
    """Test that `_merge_transcriptions_by_sentence` correctly merges transcriptions into sentences
    when the text contains Arabic question marks (؟) as terminal punctuation.
    """
    transcriber = _build_transcriber()
    data = pd.DataFrame(
        [
            {"start": "0:00:00", "end": "0:00:01", "text": "مرحبا"},
            {"start": "0:00:01", "end": "0:00:02", "text": "كيف الحال؟"},
        ]
    )

    merged = transcriber._merge_transcriptions_by_sentence(data)

    assert merged.to_dict("records") == [
        {"start": "0:00:00", "end": "0:00:02", "text": "مرحبا كيف الحال؟"}
    ]


def test_merge_transcriptions_preserves_speaker_column() -> None:
    """Test that sentence merging keeps speaker labels for diarized rows."""
    transcriber = _build_transcriber()
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

    merged = transcriber._merge_transcriptions_by_sentence(data)

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
    transcriber = _build_transcriber()
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

    merged = transcriber._merge_transcriptions_by_sentence(data)

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


def test_module_level_merge_function_matches_instance_method() -> None:
    """Test that the module-level _merge_transcriptions_by_sentence produces the same result
    as calling the instance method on WhisperTranscriber.
    """
    transcriber = _build_transcriber()
    data = pd.DataFrame(
        [
            {"start": "0:00:00", "end": "0:00:01", "text": "hello"},
            {"start": "0:00:01", "end": "0:00:02", "text": "world."},
        ]
    )
    instance_result = transcriber._merge_transcriptions_by_sentence(data)
    module_result = _merge_transcriptions_by_sentence(data.copy())
    pd.testing.assert_frame_equal(instance_result, module_result)


# ---------------------------------------------------------------------------
# Torch safe globals
# ---------------------------------------------------------------------------


def test_configure_torch_safe_globals_registers_checkpoint_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that Torch safe globals include required checkpoint classes.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture for patching.
    """
    recorded_globals: list[type] = []

    def fake_add_safe_globals(classes: list[type]) -> None:
        recorded_globals.extend(classes)

    monkeypatch.setattr(
        transcription.torch.serialization,
        "add_safe_globals",
        fake_add_safe_globals,
    )

    _configure_torch_safe_globals()

    class_names = {registered.__name__ for registered in recorded_globals}
    assert "DictConfig" in class_names
    assert "ListConfig" in class_names
    assert "TorchVersion" in class_names
    assert "Problem" in class_names
    assert "Resolution" in class_names
    assert "Specifications" in class_names


# ---------------------------------------------------------------------------
# WhisperTranscriber initialisation
# ---------------------------------------------------------------------------


def test_init_skips_diarization_model_for_single_speaker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that single-speaker runs do not arm the diarization path.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture for patching.
    """
    monkeypatch.setattr(
        WhisperTranscriber,
        "_load_audio",
        staticmethod(lambda file, sample_rate=16000: np.zeros(sample_rate)),
    )
    monkeypatch.setattr(
        WhisperTranscriber,
        "_detect_language",
        lambda self: "de",
    )
    monkeypatch.setattr(
        transcription,
        "load_mappings",
        lambda _: {"de": "German", "en": "English"},
    )

    transcriber = WhisperTranscriber(
        file_path=transcription.Path("sample.wav"),
        trg_lang="en",
        n_speakers=1,
    )

    assert transcriber.diarize_model is None


def test_init_skips_language_detection_for_silent_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Silent inputs must not trigger Whisper language detection.

    The pre-Whisper guard runs once in ``__init__`` so silent files never
    promote a Whisper model to the GPU — important on Apple Silicon, where
    sparse-tensor ops on the MPS backend would otherwise crash. The test
    builds a transcriber with all-zero audio and a fail-fast
    ``_detect_language`` stub: the assertion is the absence of the
    AssertionError that stub would raise.

    Args:
        monkeypatch (pytest.MonkeyPatch): Used to stub the audio loader and
            language detector.
    """
    monkeypatch.setattr(
        WhisperTranscriber,
        "_load_audio",
        staticmethod(lambda file, sample_rate=16000: np.zeros(sample_rate)),
    )

    def _must_not_run(_self: Any) -> str:
        raise AssertionError("language detection must be skipped for silent audio")

    monkeypatch.setattr(WhisperTranscriber, "_detect_language", _must_not_run)
    monkeypatch.setattr(
        transcription,
        "load_mappings",
        lambda _: {"de": "German", "en": "English"},
    )

    transcriber = WhisperTranscriber(
        file_path=transcription.Path("sample.wav"),
        trg_lang="en",
        n_speakers=1,
    )

    assert transcriber._speech_check[0] is False
    assert transcriber.src_lang == "en"


# ---------------------------------------------------------------------------
# Diarization
# ---------------------------------------------------------------------------


def test_diarization_requires_loaded_model() -> None:
    """Test that diarization raises a clear error when the model is missing."""
    transcriber = WhisperTranscriber.__new__(WhisperTranscriber)
    transcriber.audio = np.zeros(16000)
    transcriber.diarize_model = None
    transcriber.n_speakers = 2
    transcriber.transcription_result = {"segments": []}

    with pytest.raises(RuntimeError, match="HF_HUB_TOKEN"):
        transcriber.diarization()


def test_assign_speakers_uses_maximum_overlap() -> None:
    """Test that _assign_speakers assigns the speaker with the longest overlap."""
    transcriber = WhisperTranscriber.__new__(WhisperTranscriber)
    transcriber.transcription_result = {
        "segments": [{"start": 0.0, "end": 2.0, "text": "hello world"}]
    }

    # Build a fake diarization result using itertracks
    turn_a = SimpleNamespace(start=0.0, end=1.8)  # 1.8 s overlap
    turn_b = SimpleNamespace(start=1.8, end=2.0)  # 0.2 s overlap

    fake_annotation = MagicMock()
    fake_annotation.itertracks.return_value = [
        (turn_a, None, "SPEAKER_00"),
        (turn_b, None, "SPEAKER_01"),
    ]

    transcriber._assign_speakers(fake_annotation)

    assert transcriber.transcription_result["segments"][0]["speaker"] == "SPEAKER_00"


# ---------------------------------------------------------------------------
# Language detection (new mel-spectrogram based approach)
# ---------------------------------------------------------------------------


def test_detect_language_uses_whisper_mel_spectrogram(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that language detection uses the openai-whisper mel spectrogram API.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture for patching.
    """
    dummy_mel = torch.zeros(1, 80, 100)
    dummy_probs = {"de": 0.9, "en": 0.1}

    class DummyModel:
        def __init__(self) -> None:
            self.dims = SimpleNamespace(n_mels=80)
            self._device = torch.device("cpu")

        @property
        def device(self) -> torch.device:
            return self._device

        def to(self, device: Any) -> "DummyModel":
            self._device = (
                device if isinstance(device, torch.device) else torch.device(device)
            )
            return self

        def detect_language(self, mel: Any) -> tuple[Any, dict[str, float]]:
            return None, dummy_probs

    dummy_model = DummyModel()
    loaded_ids: list[str] = []

    def fake_load_model(name: str, device: Any = None) -> DummyModel:
        loaded_ids.append(name)
        return dummy_model

    monkeypatch.setattr(transcription.whisper, "load_model", fake_load_model)
    monkeypatch.setattr(transcription.whisper, "pad_or_trim", lambda audio: audio)
    monkeypatch.setattr(
        transcription.whisper,
        "log_mel_spectrogram",
        lambda audio, n_mels: dummy_mel,
    )
    # Start from a clean registry slot so our patched loader is called.
    REGISTRY.evict("whisper_turbo")

    t = WhisperTranscriber.__new__(WhisperTranscriber)
    t.audio = np.zeros(32000, dtype=np.float32)
    t.transcription_device = "cpu"

    detected = t._detect_language()

    assert detected == "de"
    assert loaded_ids == ["large-v3-turbo"]
    REGISTRY.evict("whisper_turbo")


# ---------------------------------------------------------------------------
# Transcription model routing via the registry
# ---------------------------------------------------------------------------


def test_transcription_acquires_task_specific_registry_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """transcription() must acquire whisper_turbo for transcribe, whisper_large for translate."""
    acquired_keys: list[str] = []
    fake_model = MagicMock()
    fake_model.transcribe.return_value = {"segments": []}

    class _FakeCtx:
        def __enter__(self) -> Any:
            return fake_model

        def __exit__(self, *exc: Any) -> None:
            return None

    class _FakeRegistry:
        def acquire(self, key: str, *, device: Any = None) -> _FakeCtx:
            acquired_keys.append(key)
            return _FakeCtx()

    monkeypatch.setattr(transcription, "REGISTRY", _FakeRegistry())
    # Disable VAD so the test reaches the Whisper model registry
    monkeypatch.setattr(transcription, "_get_vad", lambda: None)

    t = WhisperTranscriber.__new__(WhisperTranscriber)
    t.audio = np.full(16000, 0.05, dtype=np.float32)
    t.task = "translate"
    t.src_lang = "de"
    t.transcription_result = None
    # __init__ caches the speech-guard result so transcription() doesn't have
    # to recompute. Bypassing __init__ in the test means we set it manually.
    t._speech_check = (True, None)

    t.transcription()
    assert acquired_keys == ["whisper_large"]

    acquired_keys.clear()
    t.task = "transcribe"
    t.transcription()
    assert acquired_keys == ["whisper_turbo"]


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


def test_external_transcriber_transcription_populates_src_lang(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that ExternalWhisperTranscriber.transcription() populates src_lang from the response.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture for patching.
    """
    seg = SimpleNamespace(start=0.0, end=1.0, text="Hola.", no_speech_prob=0.1)
    fake_response = SimpleNamespace(segments=[seg], language="es")

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
    # The new pre-Whisper guard would otherwise decode this test file as
    # audio; stub both the loader and the guard so the test focuses on
    # src_lang propagation only.
    monkeypatch.setattr(
        transcription,
        "_load_audio_waveform",
        lambda _path: np.zeros(1, dtype=np.float32),
    )
    monkeypatch.setattr(transcription, "_audio_has_speech", lambda _audio: (True, None))

    transcriber.transcription()

    assert transcriber.src_lang == "es"
    assert transcriber.transcription_result is not None
    assert len(transcriber.transcription_result["segments"]) == 1


def test_external_transcriber_normalizes_full_language_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI returns full language names; ``src_lang`` must normalize to ISO.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture for
            patching the lazy client and the audio guard.
    """
    seg = SimpleNamespace(start=0.0, end=1.0, text="Hallo.", no_speech_prob=0.1)
    fake_response = SimpleNamespace(segments=[seg], language="german")

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
    monkeypatch.setattr(
        transcription,
        "_load_audio_waveform",
        lambda _path: np.zeros(1, dtype=np.float32),
    )
    monkeypatch.setattr(transcription, "_audio_has_speech", lambda _audio: (True, None))

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
        at this test module. Callers always stub ``_load_audio_waveform``
        before ``transcription()`` so the path is never actually decoded.
    """

    transcriber = ExternalWhisperTranscriber.__new__(ExternalWhisperTranscriber)
    transcriber.file_path = transcription.Path(__file__)
    transcriber.src_lang = None
    transcriber.task = "transcribe"
    transcriber._model_id = "whisper-1"
    transcriber._client = None
    transcriber.start_column = "start"
    transcriber.end_column = "end"
    transcriber.speaker_column = "speaker"
    transcriber.text_column = "text"
    transcriber.transcription_result = None
    return transcriber


def test_audio_has_speech_flags_digital_silence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A near-zero RMS waveform must be rejected before Silero VAD runs.

    The RMS layer is the cheap first line of defence: it fires without
    consulting Silero VAD. The test monkeypatches ``_detect_speech_vad``
    to raise if called, which would signal a regression where the RMS
    short-circuit was removed.

    Args:
        monkeypatch (pytest.MonkeyPatch): Used to fail-fast if the VAD
            model is consulted on digital silence.
    """

    def _must_not_run(_audio: np.ndarray) -> bool:
        raise AssertionError("VAD must not be consulted for digital silence")

    monkeypatch.setattr(transcription, "_detect_speech_vad", _must_not_run)
    silent = np.zeros(16000, dtype=np.float32)

    ok, reason = transcription._audio_has_speech(silent)

    assert ok is False
    assert reason is not None and "RMS" in reason


def test_audio_has_speech_rejects_non_speech_when_vad_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A loud waveform with no VAD-detected speech must be rejected.

    Simulates "energy without speech" (noise, music, tone). Ensures the
    second layer of the guard fires when :func:`load_vad_env` reports
    ``enabled=True`` (the default).

    Args:
        monkeypatch (pytest.MonkeyPatch): Used to stub ``_detect_speech_vad``
            and pin ``load_vad_env`` to the enabled state.
    """

    monkeypatch.setattr(transcription, "_detect_speech_vad", lambda _audio: False)
    monkeypatch.setattr(
        transcription, "load_vad_env", lambda: SimpleNamespace(enabled=True)
    )
    loud = np.ones(16000, dtype=np.float32) * 0.5

    ok, reason = transcription._audio_has_speech(loud)

    assert ok is False
    assert reason == "VAD detected no speech"


def test_audio_has_speech_passes_when_vad_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``VAD_ENABLED=0`` the guard must skip the Silero check.

    Operators can disable VAD via ``VAD_ENABLED=0``; the RMS layer then
    stands alone. The test asserts ``_detect_speech_vad`` is never called
    in that configuration even though the audio *would* be rejected by
    VAD (``False`` would otherwise trip the guard).

    Args:
        monkeypatch (pytest.MonkeyPatch): Disables the env toggle and
            supplies a fail-fast stub for the VAD model.
    """

    def _must_not_run(_audio: np.ndarray) -> bool:
        raise AssertionError("VAD must not be consulted when VAD_ENABLED=0")

    monkeypatch.setattr(transcription, "_detect_speech_vad", _must_not_run)
    monkeypatch.setattr(
        transcription, "load_vad_env", lambda: SimpleNamespace(enabled=False)
    )
    loud = np.ones(16000, dtype=np.float32) * 0.5

    ok, reason = transcription._audio_has_speech(loud)

    assert ok is True
    assert reason is None


def test_audio_has_speech_passes_when_vad_model_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VAD returning ``None`` (graceful fallback) must not block ingestion.

    :func:`_detect_speech_vad` returns ``None`` when the Silero model
    could not be loaded — e.g. offline machine with a cold torch-hub
    cache. In that case the guard trusts the RMS layer and lets the
    audio through rather than refusing to transcribe anything.

    Args:
        monkeypatch (pytest.MonkeyPatch): Stubs ``_detect_speech_vad`` to
            simulate the unavailable-model branch.
    """

    monkeypatch.setattr(transcription, "_detect_speech_vad", lambda _audio: None)
    monkeypatch.setattr(
        transcription, "load_vad_env", lambda: SimpleNamespace(enabled=True)
    )
    loud = np.ones(16000, dtype=np.float32) * 0.5

    ok, reason = transcription._audio_has_speech(loud)

    assert ok is True
    assert reason is None


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


def test_external_transcriber_skips_on_rms_silence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Silent audio must short-circuit the external request entirely.

    The whole point of guarding the external path is to avoid paying for
    a transcription request whose result would be hallucinated text. The
    test stubs ``_audio_has_speech`` to report RMS silence, runs
    ``transcription()``, and asserts that:

    * the method returns cleanly with ``transcription_result`` holding an
      empty segment list;
    * the OpenAI client was never dereferenced — verified by patching
      ``_get_client`` with a property that raises if accessed.

    Args:
        monkeypatch (pytest.MonkeyPatch): Overrides ``_load_audio_waveform``,
            ``_audio_has_speech``, and the client property.
    """

    transcriber = _make_external_transcriber()
    monkeypatch.setattr(
        transcription,
        "_load_audio_waveform",
        lambda _path: np.zeros(1, dtype=np.float32),
    )
    monkeypatch.setattr(
        transcription,
        "_audio_has_speech",
        lambda _audio: (False, "Audio RMS (0.000000) below silence threshold (0.01)"),
    )

    def _client_must_not_be_built(_self: Any) -> Any:
        raise AssertionError("external API must not be contacted for silent audio")

    monkeypatch.setattr(
        type(transcriber),
        "_get_client",
        property(_client_must_not_be_built),
    )

    transcriber.transcription()

    assert transcriber.transcription_result == {"segments": []}


def test_external_transcriber_skips_on_vad(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Noise-only audio (VAD negative) must also skip the external request.

    Same contract as :func:`test_external_transcriber_skips_on_rms_silence`
    but for the second guard layer. Ensures that a VAD-negative verdict
    produces an empty result and never contacts the remote API.

    Args:
        monkeypatch (pytest.MonkeyPatch): Overrides ``_load_audio_waveform``,
            ``_audio_has_speech``, and the client property.
    """

    transcriber = _make_external_transcriber()
    monkeypatch.setattr(
        transcription,
        "_load_audio_waveform",
        lambda _path: np.ones(16000, dtype=np.float32) * 0.5,
    )
    monkeypatch.setattr(
        transcription,
        "_audio_has_speech",
        lambda _audio: (False, "VAD detected no speech"),
    )

    def _client_must_not_be_built(_self: Any) -> Any:
        raise AssertionError("external API must not be contacted for noise-only audio")

    monkeypatch.setattr(
        type(transcriber),
        "_get_client",
        property(_client_must_not_be_built),
    )

    transcriber.transcription()

    assert transcriber.transcription_result == {"segments": []}


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

    The VAD guard is bypassed with a ``(True, None)`` stub so the test
    focuses exclusively on the post-filter behaviour.

    Args:
        monkeypatch (pytest.MonkeyPatch): Overrides ``_load_audio_waveform``,
            ``_audio_has_speech``, and the client property.
    """

    kept = SimpleNamespace(start=0.0, end=1.0, text="Hello", no_speech_prob=0.1)
    dropped = SimpleNamespace(
        start=1.0, end=2.0, text="hallucinated.", no_speech_prob=0.95
    )
    fake_response = SimpleNamespace(segments=[kept, dropped], language="en")

    fake_client = MagicMock()
    fake_client.audio.transcriptions.create.return_value = fake_response

    transcriber = _make_external_transcriber()
    monkeypatch.setattr(
        transcription,
        "_load_audio_waveform",
        lambda _path: np.ones(16000, dtype=np.float32) * 0.5,
    )
    monkeypatch.setattr(transcription, "_audio_has_speech", lambda _audio: (True, None))
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
        monkeypatch (pytest.MonkeyPatch): Overrides ``_load_audio_waveform``,
            ``_audio_has_speech``, and the client property.
    """

    seg = SimpleNamespace(start=0.0, end=1.0, text="Hola.")
    fake_response = SimpleNamespace(segments=[seg], language="es")

    fake_client = MagicMock()
    fake_client.audio.transcriptions.create.return_value = fake_response

    transcriber = _make_external_transcriber()
    monkeypatch.setattr(
        transcription,
        "_load_audio_waveform",
        lambda _path: np.ones(16000, dtype=np.float32) * 0.5,
    )
    monkeypatch.setattr(transcription, "_audio_has_speech", lambda _audio: (True, None))
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
