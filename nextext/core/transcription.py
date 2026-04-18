"""Audio transcription and diarization with openai-whisper."""

import os
from datetime import timedelta
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any, Optional, cast

import numpy as np
import pandas as pd  # type: ignore[import]
import torch
import whisper  # type: ignore[import]
from dotenv import load_dotenv
from loguru import logger
from openai import APIStatusError, OpenAI
from pyannote.audio import Pipeline as DiarizationPipeline  # type: ignore[import]
from nextext.utils.env_cfg import load_inference_env, load_vad_env
from nextext.utils.mappings_loader import load_mappings
from nextext.utils.model_registry import REGISTRY, ModelSpec, Strategy


def _load_optional_module(module_name: str) -> ModuleType | None:
    """Load an optional module and return None when unavailable."""
    try:
        return import_module(module_name)
    except ImportError:  # pragma: no cover - optional dependency
        return None


_omegaconf = _load_optional_module("omegaconf")

DictConfig: Any = getattr(_omegaconf, "DictConfig", None)
ListConfig: Any = getattr(_omegaconf, "ListConfig", None)

_torch_version = _load_optional_module("torch.torch_version")

TorchVersion: Any = getattr(_torch_version, "TorchVersion", None)

_pyannote_task = _load_optional_module("pyannote.audio.core.task")

Problem: Any = getattr(_pyannote_task, "Problem", None)
Resolution: Any = getattr(_pyannote_task, "Resolution", None)
Specifications: Any = getattr(_pyannote_task, "Specifications", None)


load_dotenv()
HF_HUB_TOKEN = os.getenv("HF_HUB_TOKEN", "")

LOCAL_WHISPER_MODELS: dict[str, str] = {
    "transcribe": "large-v3-turbo",
    "translate": "large-v3",
}

_WHISPER_REGISTRY_KEYS: dict[str, str] = {
    "transcribe": "whisper_turbo",
    "translate": "whisper_large",
}
_DIARIZATION_MODEL_ID: str = "pyannote/speaker-diarization-3.1"

OPENAI_WHISPER_MAX_UPLOAD_BYTES: int = 25 * 1024 * 1024

# Segments with ``no_speech_prob`` above this threshold are discarded.
# Whisper's built-in filter requires *both* high ``no_speech_prob`` and low
# ``avg_logprob``, which lets confident hallucinations on silent audio slip
# through.  Filtering on ``no_speech_prob`` alone catches them.
NO_SPEECH_THRESHOLD: float = 0.6

# Audio with an RMS energy below this value is treated as silence and skipped
# before Whisper runs.  0.01 ≈ −40 dB; normal speech sits well above 0.03.
# This catches cases where ``no_speech_prob`` filtering alone is insufficient
# because Whisper can hallucinate with low ``no_speech_prob`` on quiet noise.
SILENCE_RMS_THRESHOLD: float = 0.01

SILERO_VAD_REPO: str = "snakers4/silero-vad"

# Three-state lazy cache: False = not attempted, None = failed, tuple = ready.
_vad_cache: tuple[Any, Any] | None | bool = False


def _get_vad() -> tuple[Any, Any] | None:
    """Lazily load the Silero VAD model via ``torch.hub``.

    Returns:
        A ``(model, get_speech_timestamps)`` tuple, or ``None`` when the
        model could not be loaded (network error, missing cache, etc.).
        Failure is cached so ``torch.hub.load`` is not retried on every file.
    """
    global _vad_cache
    if _vad_cache is not False:
        return cast(Optional[tuple[Any, Any]], _vad_cache)
    try:
        model, utils = torch.hub.load(
            SILERO_VAD_REPO,
            model="silero_vad",
            trust_repo=True,
        )
        _vad_cache = (model, utils[0])  # utils[0] = get_speech_timestamps
        logger.info("Silero VAD model loaded.")
    except Exception as exc:
        logger.warning("Could not load Silero VAD ({}). Falling back to RMS-only.", exc)
        _vad_cache = None
    return cast(Optional[tuple[Any, Any]], _vad_cache)


def _detect_speech_vad(audio: np.ndarray, sample_rate: int = 16000) -> bool | None:
    """Check whether *audio* contains human speech using Silero VAD.

    Args:
        audio: Float32 waveform (mono, 16 kHz).
        sample_rate: Sample rate of *audio*.

    Returns:
        ``True`` if speech is found, ``False`` if no speech is detected,
        or ``None`` when the VAD model is unavailable (graceful fallback).
    """
    vad = _get_vad()
    if vad is None:
        return None
    model, get_speech_timestamps = vad
    tensor = torch.from_numpy(audio).float()
    timestamps = get_speech_timestamps(tensor, model, sampling_rate=sample_rate)
    return len(timestamps) > 0


def _load_audio_waveform(file_path: Path) -> np.ndarray:
    """Decode *file_path* into a float32 mono 16 kHz waveform.

    Thin wrapper over :func:`whisper.load_audio` so every caller in this
    module — both the local and the external transcriber — shares the
    exact same decoding step (same sample rate, same normalisation).

    Args:
        file_path: Path to the audio file to decode.

    Returns:
        Float32 mono waveform sampled at 16 kHz.
    """
    return whisper.load_audio(str(file_path))


def _audio_has_speech(audio: np.ndarray) -> tuple[bool, str | None]:
    """Gate an audio waveform against the pre-Whisper hallucination guards.

    Combines the two pre-transcription checks that both the local and
    external transcribers run. Keeping them in one helper ensures the two
    paths cannot drift:

    1. **RMS energy** — waveforms below :data:`SILENCE_RMS_THRESHOLD` are
       treated as digital silence and rejected without invoking Silero.
    2. **Silero VAD** — when ``VAD_ENABLED`` is set (the default, via
       :func:`load_vad_env`), waveforms that carry energy but no
       detectable human speech are rejected. VAD returning ``None``
       (graceful fallback when the model could not be loaded) is treated
       as a pass: the RMS check has already run.

    Args:
        audio: Float32 mono waveform sampled at 16 kHz (the format
            returned by :func:`_load_audio_waveform`).

    Returns:
        ``(True, None)`` when the audio should be sent to Whisper;
        ``(False, reason)`` when it should be skipped. *reason* is a
        short, log-friendly explanation.
    """
    rms = float(np.sqrt(np.mean(audio**2)))
    if rms < SILENCE_RMS_THRESHOLD:
        return (
            False,
            f"Audio RMS ({rms:.6f}) below silence threshold ({SILENCE_RMS_THRESHOLD})",
        )
    if load_vad_env().enabled:
        has_speech = _detect_speech_vad(audio)
        if has_speech is False:
            return False, "VAD detected no speech"
    return True, None


def _filter_no_speech_segments(
    segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drop segments whose ``no_speech_prob`` exceeds the threshold.

    Whisper's built-in filter is conservative (it also requires a low
    ``avg_logprob``) and lets confident hallucinations on quiet or
    silent audio through. Filtering on ``no_speech_prob`` alone catches
    those; the same threshold is applied to the local and external paths
    so downstream output is consistent regardless of provider.

    Args:
        segments: Raw Whisper segment dicts; missing ``no_speech_prob`` is
            treated as ``0.0``.

    Returns:
        The filtered segment list. When nothing is dropped the original
        list object is returned unchanged (identity preserved).
    """
    filtered = [
        seg
        for seg in segments
        if float(seg.get("no_speech_prob", 0.0)) <= NO_SPEECH_THRESHOLD
    ]
    dropped = len(segments) - len(filtered)
    if dropped:
        logger.info(
            "Dropped {}/{} segments with no_speech_prob > {}.",
            dropped,
            len(segments),
            NO_SPEECH_THRESHOLD,
        )
        return filtered
    return segments


def _configure_torch_safe_globals() -> None:
    """Register safe globals needed for Pyannote checkpoints on Torch 2.6+."""
    add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
    if add_safe_globals is None:
        return

    safe_globals: list[Any] = []

    if DictConfig is not None and ListConfig is not None:
        safe_globals.extend([DictConfig, ListConfig])
    else:
        logger.debug(
            "OmegaConf is unavailable; skipping Torch safe-global registration."
        )

    if TorchVersion is not None:
        safe_globals.append(TorchVersion)
    else:
        logger.debug(
            "TorchVersion is unavailable; skipping Torch safe-global registration."
        )

    if Problem is not None and Resolution is not None and Specifications is not None:
        safe_globals.extend([Problem, Resolution, Specifications])
    else:
        logger.debug(
            "Pyannote task classes are unavailable; skipping Torch safe-global registration."
        )

    if safe_globals:
        add_safe_globals(safe_globals)


def _seconds_to_time(seconds: float) -> str:
    """Convert seconds to a string representation of time in the format HH:MM:SS.

    Args:
        seconds (float): The number of seconds to convert.

    Returns:
        str: The string representation of time in the format HH:MM:SS.
    """
    return str(timedelta(seconds=round(seconds)))


def _ends_with_punctuation(text: str) -> bool:
    """Check if the given text ends with a sentence-ending punctuation mark.

    Args:
        text (str): The text string to check.

    Returns:
        bool: True if the text ends with a supported sentence-ending punctuation mark, otherwise False.
    """
    return text.strip().endswith((".", "!", "?", "؟", "۔"))


def _merge_transcriptions_by_sentence(
    data: pd.DataFrame,
    start_column: str = "start",
    end_column: str = "end",
    speaker_column: str = "speaker",
    text_column: str = "text",
) -> pd.DataFrame:
    """Merge transcriptions by sentences based on punctuation.

    Args:
        data (pd.DataFrame): The original DataFrame containing transcription data.
        start_column (str): Column name for segment start times.
        end_column (str): Column name for segment end times.
        speaker_column (str): Column name for speaker labels.
        text_column (str): Column name for transcribed text.

    Returns:
        pd.DataFrame: A new DataFrame with merged sentences and adjusted timestamps.
    """
    output_columns: list[str] = [start_column, end_column]
    has_speaker = speaker_column in data.columns
    if has_speaker:
        output_columns.append(speaker_column)
    output_columns.append(text_column)

    if data.empty or text_column not in data.columns:
        logger.warning("No transcription rows were available for sentence merging.")
        return pd.DataFrame(columns=pd.Index(output_columns))

    def _build_empty_row() -> dict[str, Any]:
        row: dict[str, Any] = {
            start_column: None,
            end_column: None,
            text_column: "",
        }
        if has_speaker:
            row[speaker_column] = None
        return row

    def _append_current_row(end_value: Any) -> None:
        current_text = str(current_row[text_column] or "").strip()
        if not current_text:
            return
        current_row[end_column] = end_value
        current_row[text_column] = current_text
        new_rows.append(current_row.copy())

    new_rows: list[dict[str, Any]] = []
    current_row = _build_empty_row()
    previous_end: Any = None

    for _, row in data.iterrows():
        row_speaker = row.get(speaker_column) if has_speaker else None
        current_text = str(current_row[text_column] or "").strip()
        if (
            has_speaker
            and current_text
            and current_row.get(speaker_column) != row_speaker
        ):
            _append_current_row(previous_end)
            current_row = _build_empty_row()

        if current_row.get(start_column) is None:
            current_row[start_column] = row.get(start_column)
        if has_speaker and current_row.get(speaker_column) is None:
            current_row[speaker_column] = row_speaker

        if row[text_column]:
            current_row[text_column] += row[text_column].strip() + " "

        previous_end = row.get(end_column)
        if _ends_with_punctuation(row[text_column]):
            _append_current_row(previous_end)
            current_row = _build_empty_row()

    if str(current_row[text_column] or "").strip():
        _append_current_row(previous_end)

    merged_df = pd.DataFrame(new_rows, columns=pd.Index(output_columns))
    logger.info("Transcriptions successfully merged by sentence.")
    return merged_df


_configure_torch_safe_globals()


def _load_whisper(model_id: str) -> Any:
    """Load an openai-whisper model onto CPU.

    The registry mover is responsible for moving the instance onto GPU when
    acquired, so loaders always build on CPU regardless of hardware.

    Args:
        model_id (str): openai-whisper model identifier.

    Returns:
        Any: The loaded Whisper model on CPU.
    """
    logger.info("Loading Whisper model '{}' on CPU.", model_id)
    return whisper.load_model(model_id, device="cpu")


def _move_torch_module(model: Any, device: str) -> Any:
    """Move an ``nn.Module``-like object (Whisper / pyannote) to ``device``.

    Args:
        model (Any): A model with a ``.to()`` method, e.g. a Whisper model or
            a pyannote ``Pipeline``.
        device (str): Target device string, e.g. ``"cuda"`` or ``"cpu"``.

    Returns:
        Any: The same model instance after the in-place move.
    """
    model.to(torch.device(device))
    return model


def _load_diarization_pipeline() -> DiarizationPipeline:
    """Load the pyannote speaker diarization pipeline onto CPU.

    Returns:
        DiarizationPipeline: The loaded pyannote pipeline on CPU, ready to be
            promoted to GPU via :func:`_move_torch_module`.

    Raises:
        RuntimeError: If ``HF_HUB_TOKEN`` is not set in the environment.
    """
    if not HF_HUB_TOKEN:
        raise RuntimeError(
            "Speaker diarization requires HF_HUB_TOKEN. Set it in your environment."
        )
    logger.info("Loading diarization pipeline '{}' on CPU.", _DIARIZATION_MODEL_ID)
    return DiarizationPipeline.from_pretrained(
        _DIARIZATION_MODEL_ID,
        use_auth_token=HF_HUB_TOKEN,
    )


# Whisper and pyannote use sparse-tensor ops that are not implemented on
# the Apple Silicon SparseMPS backend; PYTORCH_ENABLE_MPS_FALLBACK=1 only
# covers the regular MPS backend, so promoting these models to MPS raises
# NotImplementedError mid-move. Mark them mps_compatible=False so acquire()
# pins them to CPU on Mac while CUDA is still used where available.
REGISTRY.register(
    ModelSpec(
        name="whisper_turbo",
        loader=lambda: _load_whisper(LOCAL_WHISPER_MODELS["transcribe"]),
        mover=_move_torch_module,
        default_strategy=Strategy.OFFLOAD,
        mps_compatible=False,
    )
)
REGISTRY.register(
    ModelSpec(
        name="whisper_large",
        loader=lambda: _load_whisper(LOCAL_WHISPER_MODELS["translate"]),
        mover=_move_torch_module,
        default_strategy=Strategy.OFFLOAD,
        mps_compatible=False,
    )
)
REGISTRY.register(
    ModelSpec(
        name="diarization",
        loader=_load_diarization_pipeline,
        mover=_move_torch_module,
        default_strategy=Strategy.OFFLOAD,
        mps_compatible=False,
    )
)


class WhisperTranscriber:
    """Transcribes and optionally diarizes audio using openai-whisper and pyannote.

    All GPU-resident models (Whisper turbo, Whisper large, diarization pipeline)
    are managed through the process-wide :data:`REGISTRY` and are acquired
    lazily on first use.  Between files, call
    :func:`nextext.utils.model_registry.flush_gpu` to reclaim VRAM.

    Attributes:
        src_lang (str): Source language of the audio, resolved during ``__init__``.
        task (str): ``"transcribe"`` or ``"translate"``.
        n_speakers (int): Maximum number of speakers for diarization.
        start_column (str): DataFrame column for segment start times.
        end_column (str): DataFrame column for segment end times.
        speaker_column (str): DataFrame column for speaker labels.
        text_column (str): DataFrame column for transcribed text.
        transcription_device (str): Preferred device for inference
            (``"cuda"`` or ``"cpu"``).
        diarize_model (Any): ``True`` when diarization is available (token
            present and ``n_speakers > 1``); ``None`` otherwise.  The actual
            pipeline is acquired from the registry lazily inside
            :meth:`diarization`.
        audio (np.ndarray): Audio loaded at 16 kHz by whisper.
        transcription_result (Optional[dict[str, Any]]): Raw output populated
            by :meth:`transcription`.
        df (Optional[pd.DataFrame]): Final DataFrame populated by
            :meth:`transcript_output`.
    """

    def __init__(
        self,
        file_path: Path,
        trg_lang: str,
        src_lang: str | None = None,
        task: str = "transcribe",
        n_speakers: int = 1,
        start_column: str = "start",
        end_column: str = "end",
        speaker_column: str = "speaker",
        text_column: str = "text",
        whisper_language_file: str = "whisper_languages.json",
    ) -> None:
        """Initialize the WhisperTranscriber object.

        Args:
            file_path (Path): The path to the input file.
            trg_lang (str): The target language for translation.
            src_lang (str, optional): The source language of the file. Defaults to None, which triggers language detection.
            task (str): Indicates whether the task is transcription or translation. Defaults to "transcribe".
            n_speakers (int): The maximum number of speakers to identify in the audio. Defaults to 1.
            start_column (str): The text column with the starting timestamp. Defaults to "start".
            end_column (str): The text column with the ending timestamp. Defaults to "end".
            speaker_column (str): The text column with the speaker information. Defaults to "speaker".
            text_column (str): The text column where the result is stored. Defaults to "text".
            whisper_language_file (str): Path to the Whisper language mapping file.
        """
        self.n_speakers = n_speakers
        self.start_column = start_column
        self.end_column = end_column
        self.speaker_column = speaker_column
        self.text_column = text_column

        self.transcription_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.audio = self._load_audio(file_path)
        # Run the silence/VAD guard once, up-front. If the file has no speech
        # we skip language detection entirely: there is no point promoting
        # Whisper to its accelerator for audio we know we will not transcribe.
        self._speech_check: tuple[bool, str | None] = _audio_has_speech(self.audio)

        whisper_languages = load_mappings(whisper_language_file)
        if self._speech_check[0]:
            det_lang = self._detect_language() or "en"
        else:
            logger.info("{}; skipping language detection.", self._speech_check[1])
            det_lang = src_lang or "en"
        self.src_lang = src_lang if src_lang in whisper_languages.keys() else det_lang
        self.task = "transcribe" if det_lang == trg_lang else task
        logger.info("Using language '{}' for task '{}'", self.src_lang, self.task)

        self.diarize_model: Any = None
        if self.n_speakers > 1:
            if not HF_HUB_TOKEN:
                logger.warning(
                    "No Hugging Face token provided. Speaker diarization will be unavailable."
                )
            else:
                # Sentinel: diarization is available; pipeline is acquired lazily
                # via REGISTRY.acquire("diarization") inside diarization().
                self.diarize_model = True

        self.transcription_result: Optional[dict[str, Any]] = None
        self.df: pd.DataFrame | None = None

    @staticmethod
    def _load_audio(file: Path, sample_rate: int = 16000) -> np.ndarray:
        """Load the audio file as a numpy array using the whisper library.

        Thin delegate over :func:`_load_audio_waveform` kept so existing
        callers and tests that reach into ``WhisperTranscriber._load_audio``
        keep working.

        Args:
            file (Path): The path to the audio file.
            sample_rate (int): Unused; whisper always resamples to 16000 Hz.

        Returns:
            np.ndarray: The loaded audio as a float32 array at 16 kHz.
        """
        return _load_audio_waveform(file)

    def _detect_language(
        self, duration_sec: float = 30.0, sample_rate: int = 16000
    ) -> str | None:
        """Detect the spoken language using ``large-v3-turbo``.

        The Whisper model is acquired via the process-wide model registry and
        released as soon as detection finishes. Under the OFFLOAD strategy it
        stays cached on CPU so a subsequent transcription run can re-promote
        it to GPU without a fresh disk load.

        Args:
            duration_sec (float): Number of seconds from the start of the audio to use for detection.
            sample_rate (int): Sample rate for the audio processing. Defaults to 16000.

        Returns:
            str | None: Detected language code (e.g., ``"en"``).

        Raises:
            RuntimeError: If language detection fails.
        """
        sample_frames = int(duration_sec * sample_rate)
        audio_clip = whisper.pad_or_trim(self.audio[:sample_frames])
        with REGISTRY.acquire("whisper_turbo") as model:
            mel = whisper.log_mel_spectrogram(audio_clip, n_mels=model.dims.n_mels).to(
                model.device
            )
            _, probs = model.detect_language(mel)
            detected_lang = max(probs, key=probs.get)

        if not detected_lang:
            raise RuntimeError("Language detection failed.")

        logger.info("Detected language: {}", detected_lang)
        return detected_lang

    def transcription(self) -> None:
        """Run the transcription process on the loaded audio file using openai-whisper.

        Three pre-/post-transcription guards prevent Whisper hallucinations.
        Layers 1 & 2 run once in :meth:`__init__` (so silent files never
        promote a model to GPU); the cached verdict is consulted here:

        1. **RMS energy** — audio below :data:`SILENCE_RMS_THRESHOLD` is
           digital silence and is skipped instantly.
        2. **Silero VAD** — when enabled via ``VAD_ENABLED``, audio that
           contains energy but no human speech is skipped before Whisper
           runs.
        3. **no_speech_prob** — segments whose ``no_speech_prob`` exceeds
           :data:`NO_SPEECH_THRESHOLD` are discarded after transcription.
        """
        # Layers 1 & 2: RMS + Silero VAD (computed once in __init__).
        has_speech, skip_reason = self._speech_check
        if not has_speech:
            logger.info("{}; skipping transcription.", skip_reason)
            self.transcription_result = {"segments": []}
            return

        # Layer 3: Whisper transcription + no_speech_prob filter
        key = _WHISPER_REGISTRY_KEYS.get(self.task, "whisper_turbo")
        try:
            with REGISTRY.acquire(key) as model:
                result = model.transcribe(
                    self.audio, task=self.task, language=self.src_lang
                )
            self.transcription_result = {
                "segments": _filter_no_speech_segments(result["segments"])
            }
        except Exception as e:
            logger.error("Error during transcription: {}", e, exc_info=True)
            raise

    def _assign_speakers(self, diarization_result: Any) -> None:
        """Assign speaker labels to transcription segments based on maximum overlap.

        Args:
            diarization_result: A pyannote Annotation object from the diarization pipeline.
        """
        if self.transcription_result is None:
            return
        for segment in self.transcription_result["segments"]:
            seg_start: float = segment["start"]
            seg_end: float = segment["end"]
            speaker_durations: dict[str, float] = {}
            for turn, _, speaker in diarization_result.itertracks(yield_label=True):
                overlap_start = max(seg_start, turn.start)
                overlap_end = min(seg_end, turn.end)
                if overlap_end > overlap_start:
                    speaker_durations[speaker] = speaker_durations.get(speaker, 0.0) + (
                        overlap_end - overlap_start
                    )
            if speaker_durations:
                segment["speaker"] = max(
                    speaker_durations, key=lambda s: speaker_durations[s]
                )

    def diarization(self) -> None:
        """Perform speaker diarization on the audio file and assign speaker labels to segments.

        Raises:
            RuntimeError: If diarization is requested but the diarization model is not available.
        """
        if self.n_speakers <= 1:
            logger.info("Skipping diarization as only one speaker is specified.")
            return
        if self.diarize_model is None:
            raise RuntimeError(
                "Speaker diarization requires a valid HF_HUB_TOKEN and "
                "accepted pyannote model access."
            )
        try:
            waveform = torch.tensor(self.audio).unsqueeze(0)
            with REGISTRY.acquire("diarization") as pipeline:
                diarize_result = pipeline(
                    {"waveform": waveform, "sample_rate": 16000},
                    max_speakers=self.n_speakers,
                )
            self._assign_speakers(diarize_result)
        except Exception as e:
            logger.error("Error during diarization: {}", e, exc_info=True)
            raise

    @staticmethod
    def _ends_with_punctuation(text: str) -> bool:
        return _ends_with_punctuation(text)

    @staticmethod
    def _seconds_to_time(seconds: float) -> str:
        return _seconds_to_time(seconds)

    def _merge_transcriptions_by_sentence(self, data: pd.DataFrame) -> pd.DataFrame:
        return _merge_transcriptions_by_sentence(
            data,
            self.start_column,
            self.end_column,
            self.speaker_column,
            self.text_column,
        )

    def transcript_output(self) -> pd.DataFrame:
        """Get the transcription result as a DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the transcription results.

        Raises:
            ValueError: If the transcription result is not available or transcription has not been run.
        """
        if (
            self.transcription_result is None
            or "segments" not in self.transcription_result
        ):
            raise ValueError(
                "Transcription result is not available. Run transcription first."
            )

        segments = []
        has_speaker = any(
            "speaker" in item for item in self.transcription_result["segments"]
        )
        for item in self.transcription_result["segments"]:
            row = [
                _seconds_to_time(item["start"]),
                _seconds_to_time(item["end"]),
            ]
            if has_speaker:
                row.append(item.get("speaker", "Unknown"))
            row.append(item["text"])
            segments.append(row)

        columns: list[str] = [self.start_column, self.end_column]
        if has_speaker:
            columns.append(self.speaker_column)
        columns.append(self.text_column)

        df = pd.DataFrame(segments, columns=pd.Index(columns))
        if self.n_speakers <= 1 and has_speaker:
            df.drop(self.speaker_column, axis=1, inplace=True)
        self.df = self._merge_transcriptions_by_sentence(df)
        return self.df


class ExternalWhisperTranscriber:
    """Transcribe audio via an external OpenAI-compatible Whisper API.

    Diarization is not supported; n_speakers is silently ignored.

    Attributes:
        file_path (Path): Path to the audio file.
        src_lang (str | None): Source language code; populated from API response if not provided.
        task (str): Task type, "transcribe" or "translate".

    Methods:
        transcription(): Call the external API and store segment results.
        transcript_output(): Return the transcription result as a DataFrame.
    """

    def __init__(
        self,
        file_path: Path,
        trg_lang: str,
        src_lang: str | None = None,
        model_id: str = "whisper-1",
        task: str = "transcribe",
        start_column: str = "start",
        end_column: str = "end",
        speaker_column: str = "speaker",
        text_column: str = "text",
    ) -> None:
        """Initialize the ExternalWhisperTranscriber.

        Args:
            file_path (Path): Path to the audio file.
            trg_lang (str): Accepted for interface compatibility; not forwarded to the API.
            src_lang (str | None): Source language code. Defaults to None (API auto-detects).
            model_id (str): Model name to pass to the external API. Defaults to "whisper-1".
            task (str): "transcribe" or "translate". Defaults to "transcribe".
            start_column (str): DataFrame column for segment start times.
            end_column (str): DataFrame column for segment end times.
            speaker_column (str): Kept for interface compatibility; not used.
            text_column (str): DataFrame column for transcribed text.
        """
        self.file_path = file_path
        self.src_lang = src_lang
        self.task = task
        self._model_id = model_id
        self.start_column = start_column
        self.end_column = end_column
        self.speaker_column = speaker_column
        self.text_column = text_column
        self.transcription_result: Optional[dict[str, Any]] = None
        self._client: Any = None

    @property
    def _get_client(self) -> Any:
        """Lazily create the OpenAI-compatible client from environment variables."""
        if self._client is None:
            client_kwargs: dict[str, Any] = {"api_key": os.getenv("OPENAI_API_KEY", "")}
            base_url = os.getenv("OPENAI_API_BASE", "").rstrip("/")
            if base_url:
                client_kwargs["base_url"] = base_url
            self._client = OpenAI(**client_kwargs)
        return self._client

    def transcription(self) -> None:
        """Call the external Whisper API and store the segment results.

        The same three-layer hallucination guard the local transcriber
        uses is applied here:

        1. **RMS energy + Silero VAD** (see :func:`_audio_has_speech`) —
           the audio is decoded locally once and the guard runs before
           any remote request, so silent / noise-only files never reach
           the paid endpoint.
        2. **no_speech_prob post-filter** — segments whose
           ``no_speech_prob`` exceeds :data:`NO_SPEECH_THRESHOLD` are
           dropped from the returned payload.

        Whisper's built-in ``translate`` task always targets English and
        is exposed by OpenAI-compatible servers as a separate
        ``/v1/audio/translations`` endpoint. ``task`` itself is not
        accepted on either endpoint.
        """
        audio = _load_audio_waveform(self.file_path)
        has_speech, skip_reason = _audio_has_speech(audio)
        if not has_speech:
            logger.warning(
                "{} for {}; skipping external transcription request.",
                skip_reason,
                self.file_path.name,
            )
            self.transcription_result = {"segments": []}
            return

        client = self._get_client
        file_size = self.file_path.stat().st_size
        logger.info(
            "External Whisper request: model='{}' task='{}' language='{}' file='{}' size={}B",
            self._model_id,
            self.task,
            self.src_lang,
            self.file_path.name,
            file_size,
        )
        provider = load_inference_env().provider
        if provider == "openai" and file_size > OPENAI_WHISPER_MAX_UPLOAD_BYTES:
            size_mb = file_size / (1024 * 1024)
            limit_mb = OPENAI_WHISPER_MAX_UPLOAD_BYTES / (1024 * 1024)
            raise ValueError(
                f"Audio file '{self.file_path.name}' is {size_mb:.1f} MB, "
                f"which exceeds OpenAI's {limit_mb:.0f} MB Whisper upload limit. "
                "Compress or split the file before retrying, or switch "
                "INFERENCE_PROVIDER to 'ollama' (local) or 'vllm' (self-hosted, "
                "no hard cap)."
            )
        try:
            with open(self.file_path, "rb") as f:
                if self.task == "translate":
                    response = client.audio.translations.create(
                        model=self._model_id,
                        file=f,
                        response_format="verbose_json",
                    )
                else:
                    kwargs: dict[str, Any] = {
                        "model": self._model_id,
                        "file": f,
                        "response_format": "verbose_json",
                        "timestamp_granularities": ["segment"],
                    }
                    if self.src_lang:
                        kwargs["language"] = self.src_lang
                    response = client.audio.transcriptions.create(**kwargs)
        except APIStatusError as exc:
            body = getattr(exc, "response", None)
            body_text = body.text if body is not None else ""
            logger.error(
                "External Whisper API error {}: {}",
                exc.status_code,
                body_text or exc.message,
            )
            raise
        raw_segments = [
            {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "no_speech_prob": float(getattr(seg, "no_speech_prob", 0.0) or 0.0),
            }
            for seg in response.segments
        ]
        segments = _filter_no_speech_segments(raw_segments)
        self.transcription_result = {"segments": segments}
        if self.src_lang is None:
            self.src_lang = getattr(response, "language", None)
        logger.info(
            "External transcription complete: {} segments, language={}",
            len(segments),
            self.src_lang,
        )

    def transcript_output(self) -> pd.DataFrame:
        """Get the external transcription result as a DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the transcription results.

        Raises:
            ValueError: If transcription has not been run yet.
        """
        if (
            self.transcription_result is None
            or "segments" not in self.transcription_result
        ):
            raise ValueError(
                "Transcription result is not available. Run transcription first."
            )

        rows = []
        for item in self.transcription_result["segments"]:
            rows.append(
                [
                    _seconds_to_time(item["start"]),
                    _seconds_to_time(item["end"]),
                    item["text"],
                ]
            )

        df = pd.DataFrame(
            rows,
            columns=pd.Index([self.start_column, self.end_column, self.text_column]),
        )
        return _merge_transcriptions_by_sentence(
            df,
            self.start_column,
            self.end_column,
            self.speaker_column,
            self.text_column,
        )
