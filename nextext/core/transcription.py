"""Audio transcription with openai-whisper."""

import os
from datetime import timedelta
from pathlib import Path as Path  # re-exported for tests that monkeypatch nextext.core.transcription.Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch as torch
import whisper as whisper
from dotenv import load_dotenv
from loguru import logger
from openai import APIStatusError, OpenAI

from nextext.utils.env_cfg import load_inference_env, load_vad_env
from nextext.utils.mappings_loader import load_mappings
from nextext.utils.model_registry import REGISTRY, ModelSpec, Strategy

load_dotenv()

# Whisper always transcribes; translation is handled downstream by the LLM
# (``TEXT_MODEL``), so a single turbo checkpoint covers both language
# detection and transcription. The heavier ``large-v3`` translate model has
# been retired along with Whisper's built-in translate task.
LOCAL_WHISPER_MODEL: str = "large-v3-turbo"

OPENAI_WHISPER_MAX_UPLOAD_BYTES: int = 25 * 1024 * 1024

# Segments with ``no_speech_prob`` above this threshold are discarded.
# Whisper's built-in filter requires *both* high ``no_speech_prob`` and low
# ``avg_logprob``, which lets confident hallucinations on silent audio slip
# through.  Filtering on ``no_speech_prob`` alone catches them.
NO_SPEECH_THRESHOLD: float = 0.6

# Audio with an RMS energy below this value is treated as silence and skipped
# before Whisper runs. 0.01 ~ -40 dB; normal speech sits well above 0.03.
# This catches cases where ``no_speech_prob`` filtering alone is insufficient
# because Whisper can hallucinate with low ``no_speech_prob`` on quiet noise.
SILENCE_RMS_THRESHOLD: float = 0.01

SILERO_VAD_REPO: str = "snakers4/silero-vad"

# Three-state lazy cache: False = not attempted, None = failed, tuple = ready.
_vad_cache: tuple[Any, Any] | None | bool = False


def _normalize_whisper_language(value: str | None) -> str | None:
    """Coerce a Whisper API ``language`` field to an ISO 639-1 code.

    OpenAI's ``/v1/audio/transcriptions`` endpoint returns the lowercased full
    language name (e.g. ``"german"``), while the local ``openai-whisper``
    package returns the ISO code (e.g. ``"de"``). Downstream consumers
    (spaCy model selection, language pickers) expect the ISO form, so
    normalize here using ``whisper_languages.json`` as the source of truth.

    Args:
        value (str | None): The raw ``language`` field from the Whisper
            response, either an ISO code or a full English language name.

    Returns:
        str | None: The ISO 639-1/2 code if ``value`` matches a known Whisper
        language name; the original ``value`` if it was already an ISO code,
        empty, ``None``, or unrecognized.
    """
    if not value:
        return value
    if len(value) <= 3 and value.islower():
        return value
    code_to_name = load_mappings("whisper_languages.json")
    name_to_code = {name.lower(): code for code, name in code_to_name.items()}
    return name_to_code.get(value.lower(), value)


def _get_vad() -> tuple[Any, Any] | None:
    """Lazily load the Silero VAD model via ``torch.hub``.

    Failure is cached so ``torch.hub.load`` is not retried on every file.

    Returns:
        tuple[Any, Any] | None: A ``(model, get_speech_timestamps)`` tuple,
            or ``None`` when the model could not be loaded (network error,
            missing cache, etc.).
    """
    global _vad_cache
    if _vad_cache is not False:
        return cast(tuple[Any, Any] | None, _vad_cache)
    try:
        model, utils = torch.hub.load(  # type: ignore[no-untyped-call]
            SILERO_VAD_REPO,
            model="silero_vad",
            trust_repo=True,
        )
        _vad_cache = (model, utils[0])  # utils[0] = get_speech_timestamps
        logger.info("Silero VAD model loaded.")
    except Exception as exc:
        logger.warning("Could not load Silero VAD ({}). Falling back to RMS-only.", exc)
        _vad_cache = None
    return _vad_cache


def _detect_speech_vad(audio: np.ndarray, sample_rate: int = 16000) -> bool | None:
    """Check whether ``audio`` contains human speech using Silero VAD.

    Args:
        audio (np.ndarray): Float32 waveform (mono, 16 kHz).
        sample_rate (int): Sample rate of ``audio``.

    Returns:
        bool | None: ``True`` if speech is found, ``False`` if no speech is
            detected, or ``None`` when the VAD model is unavailable
            (graceful fallback).
    """
    vad = _get_vad()
    if vad is None:
        return None
    model, get_speech_timestamps = vad
    tensor = torch.from_numpy(audio).float()
    timestamps = get_speech_timestamps(tensor, model, sampling_rate=sample_rate)
    return len(timestamps) > 0


def _load_audio_waveform(file_path: Path) -> np.ndarray:
    """Decode ``file_path`` into a float32 mono 16 kHz waveform.

    Thin wrapper over :func:`whisper.load_audio` so every caller in this
    module — both the local and the external transcriber — shares the
    exact same decoding step (same sample rate, same normalisation).

    Args:
        file_path (Path): Path to the audio file to decode.

    Returns:
        np.ndarray: Float32 mono waveform sampled at 16 kHz.
    """
    return cast(np.ndarray, whisper.load_audio(str(file_path)))


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
        audio (np.ndarray): Float32 mono waveform sampled at 16 kHz (the
            format returned by :func:`_load_audio_waveform`).

    Returns:
        tuple[bool, str | None]: ``(True, None)`` when the audio should be
            sent to Whisper; ``(False, reason)`` when it should be skipped.
            ``reason`` is a short, log-friendly explanation.
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
        segments (list[dict[str, Any]]): Raw Whisper segment dicts; missing
            ``no_speech_prob`` is treated as ``0.0``.

    Returns:
        list[dict[str, Any]]: The filtered segment list. When nothing is
            dropped the original list object is returned unchanged (identity
            preserved).
    """
    filtered = [seg for seg in segments if float(seg.get("no_speech_prob", 0.0)) <= NO_SPEECH_THRESHOLD]
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
    return text.strip().endswith((".", "!", "?", "؟", "۔"))  # noqa: RUF001 - Arabic full stop


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
        if has_speaker and current_text and current_row.get(speaker_column) != row_speaker:
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
    """Move an ``nn.Module``-like object (the Whisper model) to ``device``.

    Args:
        model (Any): A model with a ``.to()`` method, e.g. a Whisper model.
        device (str): Target device string, e.g. ``"cuda"`` or ``"cpu"``.

    Returns:
        Any: The same model instance after the in-place move.
    """
    model.to(torch.device(device))
    return model


# Whisper uses sparse-tensor ops that are not implemented on the Apple Silicon
# SparseMPS backend; PYTORCH_ENABLE_MPS_FALLBACK=1 only covers the regular MPS
# backend, so promoting the model to MPS raises NotImplementedError mid-move.
# Mark it mps_compatible=False so acquire() pins it to CPU on Mac while CUDA is
# still used where available.
REGISTRY.register(
    ModelSpec(
        name="whisper_turbo",
        loader=lambda: _load_whisper(LOCAL_WHISPER_MODEL),
        mover=_move_torch_module,
        default_strategy=Strategy.OFFLOAD,
        mps_compatible=False,
    )
)


class WhisperTranscriber:
    """Transcribes audio using openai-whisper.

    The GPU-resident Whisper turbo checkpoint is managed through the
    process-wide :data:`REGISTRY` and acquired lazily on first use.  Between
    files, call :func:`nextext.utils.model_registry.flush_gpu` to reclaim VRAM.

    Whisper always transcribes; translation to the target language is handled
    downstream by the LLM (``TEXT_MODEL``), so this class no longer takes a
    ``task`` and never loads the retired ``large-v3`` translate model.
    Diarization is no longer performed in-process: the pipeline labels speakers
    out-of-band via the ``/diarize`` service (see
    :func:`nextext.core.diarization.diarize_file`). ``n_speakers`` is retained
    only so :meth:`transcript_output` knows whether to keep a speaker column.

    Attributes:
        src_lang (str): Source language of the audio, resolved during ``__init__``.
        n_speakers (int): Maximum speaker count requested. When greater than 1,
            speaker labels attached upstream by the pipeline are preserved by
            :meth:`transcript_output`; a value of 1 drops any speaker column.
        start_column (str): DataFrame column for segment start times.
        end_column (str): DataFrame column for segment end times.
        speaker_column (str): DataFrame column for speaker labels.
        text_column (str): DataFrame column for transcribed text.
        transcription_device (str): Preferred device for inference
            (``"cuda"`` or ``"cpu"``).
        audio (np.ndarray): Audio loaded at 16 kHz by whisper.
        transcription_result (Optional[dict[str, Any]]): Raw output populated
            by :meth:`transcription`.
        df (Optional[pd.DataFrame]): Final DataFrame populated by
            :meth:`transcript_output`.
    """

    def __init__(
        self,
        file_path: Path,
        src_lang: str | None = None,
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
            src_lang (str, optional): The source language of the file. Defaults to None
                (triggers language detection).
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
        logger.info("Using language '{}' for transcription.", self.src_lang)

        self.transcription_result: dict[str, Any] | None = None
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

    def _detect_language(self, duration_sec: float = 30.0, sample_rate: int = 16000) -> str | None:
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
            mel = whisper.log_mel_spectrogram(audio_clip, n_mels=model.dims.n_mels).to(model.device)
            _, probs = model.detect_language(mel)
            detected_lang = max(probs, key=probs.get)

        if not detected_lang:
            raise RuntimeError("Language detection failed.")

        logger.info("Detected language: {}", detected_lang)
        return cast(str | None, detected_lang)

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
        try:
            with REGISTRY.acquire("whisper_turbo") as model:
                result = model.transcribe(self.audio, task="transcribe", language=self.src_lang)
            self.transcription_result = {"segments": _filter_no_speech_segments(result["segments"])}
        except Exception as e:
            logger.error("Error during transcription: {}", e, exc_info=True)
            raise

    @staticmethod
    def _ends_with_punctuation(text: str) -> bool:
        """Delegate to the module-level sentence-terminator check.

        Args:
            text (str): The text string to check.

        Returns:
            bool: ``True`` if ``text`` ends with sentence-ending punctuation.
        """
        return _ends_with_punctuation(text)

    @staticmethod
    def _seconds_to_time(seconds: float) -> str:
        """Delegate to the module-level ``HH:MM:SS`` formatter.

        Args:
            seconds (float): The number of seconds to convert.

        Returns:
            str: The formatted ``HH:MM:SS`` string.
        """
        return _seconds_to_time(seconds)

    def _merge_transcriptions_by_sentence(self, data: pd.DataFrame) -> pd.DataFrame:
        """Merge transcription rows into sentences using instance column names.

        Args:
            data (pd.DataFrame): The original DataFrame with per-segment rows.

        Returns:
            pd.DataFrame: A new DataFrame with rows merged by sentence.
        """
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
        if self.transcription_result is None or "segments" not in self.transcription_result:
            raise ValueError("Transcription result is not available. Run transcription first.")

        segments = []
        has_speaker = any("speaker" in item for item in self.transcription_result["segments"])
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

    Diarization is not supported; n_speakers is silently ignored. The
    transcriber only ever calls ``/v1/audio/transcriptions``; translation to
    the target language is handled downstream by the LLM (``TEXT_MODEL``).
    Whisper's ``/v1/audio/translations`` endpoint is deliberately not used, as
    it always emits English and is not served by every OpenAI-compatible
    backend (vLLM returns 404 for it).

    Attributes:
        file_path (Path): Path to the audio file.
        src_lang (str | None): Source language code; populated from API response if not provided.

    Methods:
        transcription(): Call the external API and store segment results.
        transcript_output(): Return the transcription result as a DataFrame.
    """

    def __init__(
        self,
        file_path: Path,
        src_lang: str | None = None,
        model_id: str = "whisper-1",
        start_column: str = "start",
        end_column: str = "end",
        speaker_column: str = "speaker",
        text_column: str = "text",
    ) -> None:
        """Initialize the ExternalWhisperTranscriber.

        Args:
            file_path (Path): Path to the audio file.
            src_lang (str | None): Source language code. Defaults to None (API auto-detects).
            model_id (str): Model name to pass to the external API. Defaults to "whisper-1".
            start_column (str): DataFrame column for segment start times.
            end_column (str): DataFrame column for segment end times.
            speaker_column (str): Kept for interface compatibility; not used.
            text_column (str): DataFrame column for transcribed text.
        """
        self.file_path = file_path
        self.src_lang = src_lang
        self._model_id = model_id
        self.start_column = start_column
        self.end_column = end_column
        self.speaker_column = speaker_column
        self.text_column = text_column
        self.transcription_result: dict[str, Any] | None = None
        self._client: Any = None

    @property
    def _get_client(self) -> Any:
        """Lazily create the OpenAI-compatible client from environment variables.

        Returns:
            Any: The cached OpenAI client instance.
        """
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

        Only the ``/v1/audio/transcriptions`` endpoint is called. Whisper's
        ``/v1/audio/translations`` endpoint is not used: it always emits
        English and is not served by every OpenAI-compatible backend.
        Translation to the target language happens downstream in the LLM.
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
            "External Whisper request: model='{}' language='{}' file='{}' size={}B",
            self._model_id,
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
            self.src_lang = _normalize_whisper_language(getattr(response, "language", None))
        logger.info(
            "External transcription complete: {} segments, language={}",
            len(segments),
            self.src_lang,
        )

    def transcript_output(self) -> pd.DataFrame:
        """Get the external transcription result as a DataFrame.

        A ``speaker`` column is emitted only when the segments carry speaker
        labels — these are added out-of-band by the pipeline after a successful
        ``/diarize`` call (the external API itself never returns speakers).

        Returns:
            pd.DataFrame: A DataFrame with ``start``/``end``/``text`` columns,
                plus a ``speaker`` column when diarization labelled the segments.

        Raises:
            ValueError: If transcription has not been run yet.
        """
        if self.transcription_result is None or "segments" not in self.transcription_result:
            raise ValueError("Transcription result is not available. Run transcription first.")

        has_speaker = any("speaker" in item for item in self.transcription_result["segments"])
        rows = []
        for item in self.transcription_result["segments"]:
            row = [
                _seconds_to_time(item["start"]),
                _seconds_to_time(item["end"]),
            ]
            if has_speaker:
                row.append(item.get("speaker", "Unknown"))
            row.append(item["text"])
            rows.append(row)

        columns: list[str] = [self.start_column, self.end_column]
        if has_speaker:
            columns.append(self.speaker_column)
        columns.append(self.text_column)

        df = pd.DataFrame(rows, columns=pd.Index(columns))
        return _merge_transcriptions_by_sentence(
            df,
            self.start_column,
            self.end_column,
            self.speaker_column,
            self.text_column,
        )
