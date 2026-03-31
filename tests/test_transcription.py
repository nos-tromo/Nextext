"""Tests for the WhisperX transcription module."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nextext.modules import transcription
from nextext.modules.transcription import (
    TRANSCRIPTION_VAD_METHOD,
    WhisperTranscriber,
    _configure_torch_safe_globals,
)


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


def test_configure_torch_safe_globals_registers_checkpoint_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that Torch safe globals include required checkpoint classes."""
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


def test_init_skips_diarization_model_for_single_speaker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that single-speaker runs do not load the diarization pipeline.
    
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
    monkeypatch.setattr(
        WhisperTranscriber,
        "_load_transcription_model",
        lambda self, model_id, whisper_model_file: (SimpleNamespace(), None, None),
    )

    def fail_load_diarization_model(
        self,
        auth_token: str,
    ) -> None:
        """Fail when diarization loading is attempted unexpectedly.
        
        Args:
            auth_token (str): The authentication token for model loading.

        Raises:
            AssertionError: Always raised to indicate that diarization loading should not occur.
        """
        raise AssertionError(
            "_load_diarization_model should not be called for one speaker"
        )

    monkeypatch.setattr(
        WhisperTranscriber,
        "_load_diarization_model",
        fail_load_diarization_model,
    )

    transcriber = WhisperTranscriber(
        file_path=transcription.Path("sample.wav"),
        trg_lang="en",
        n_speakers=1,
    )

    assert transcriber.diarize_model is None


def test_diarization_requires_loaded_model() -> None:
    """Test that diarization raises a clear error when the model is missing."""
    transcriber = WhisperTranscriber.__new__(WhisperTranscriber)
    transcriber.audio = np.zeros(16000)
    transcriber.diarize_model = None
    transcriber.n_speakers = 2
    transcriber.transcription_result = {"segments": []}

    with pytest.raises(RuntimeError, match="HF_HUB_TOKEN"):
        transcriber.diarization()


def test_detect_language_uses_silero_vad(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that language detection avoids Pyannote VAD model loading."""
    recorded_kwargs: dict[str, object] = {}

    class DummyModel:
        """Minimal WhisperX model stub for language detection."""

        def transcribe(self, audio: np.ndarray) -> dict[str, str]:
            """Return a fixed detected language."""
            return {"language": "de"}

    def fake_load_model(**kwargs: object) -> DummyModel:
        """Capture WhisperX model-loading arguments."""
        recorded_kwargs.update(kwargs)
        return DummyModel()

    monkeypatch.setattr(transcription.whisperx, "load_model", fake_load_model)

    transcriber = WhisperTranscriber.__new__(WhisperTranscriber)
    transcriber.audio = np.zeros(32000, dtype=np.float32)
    transcriber.transcription_device = "cpu"

    detected_language = transcriber._detect_language()

    assert detected_language == "de"
    assert recorded_kwargs["vad_method"] == TRANSCRIPTION_VAD_METHOD


def test_load_transcription_model_uses_silero_vad(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that transcription model loading avoids Pyannote VAD."""
    recorded_kwargs: dict[str, object] = {}

    def fake_load_model(**kwargs: object) -> SimpleNamespace:
        """Capture WhisperX model-loading arguments."""
        recorded_kwargs.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(
        transcription,
        "load_mappings",
        lambda _: {"default_transcribe": "small"},
    )
    monkeypatch.setattr(transcription.whisperx, "load_model", fake_load_model)
    monkeypatch.setattr(
        transcription.whisperx,
        "load_align_model",
        lambda **kwargs: (None, None),
    )

    transcriber = WhisperTranscriber.__new__(WhisperTranscriber)
    transcriber.task = "transcribe"
    transcriber.src_lang = "de"
    transcriber.transcription_device = "cpu"

    transcribe_model, align_model, align_metadata = (
        transcriber._load_transcription_model("default", "unused.json")
    )

    assert transcribe_model is not None
    assert align_model is None
    assert align_metadata is None
    assert recorded_kwargs["vad_method"] == TRANSCRIPTION_VAD_METHOD
