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

    t = WhisperTranscriber.__new__(WhisperTranscriber)
    t.audio = np.zeros(16000, dtype=np.float32)
    t.task = "translate"
    t.src_lang = "de"
    t.transcription_result = None

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
    seg = SimpleNamespace(start=0.0, end=1.0, text="Hola.")
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

    transcriber.transcription()

    assert transcriber.src_lang == "es"
    assert transcriber.transcription_result is not None
    assert len(transcriber.transcription_result["segments"]) == 1
