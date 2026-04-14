"""Audio transcription and diarization with openai-whisper."""

import gc
import os
from datetime import timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import whisper
from dotenv import load_dotenv
from loguru import logger
from pyannote.audio import Pipeline as DiarizationPipeline

from nextext.utils.mappings_loader import load_mappings


load_dotenv()
HF_HUB_TOKEN = os.getenv("HF_HUB_TOKEN", "")

LOCAL_WHISPER_MODELS: dict[str, str] = {
    "transcribe": "large-v3-turbo",
    "translate": "large-v3",
}

OPENAI_WHISPER_MAX_UPLOAD_BYTES: int = 25 * 1024 * 1024


def _configure_torch_safe_globals() -> None:
    """Register safe globals needed for Pyannote checkpoints on Torch 2.6+."""
    add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
    if add_safe_globals is None:
        return

    safe_globals: list[Any] = []

    try:
        from omegaconf import DictConfig, ListConfig
    except ImportError:
        logger.debug(
            "OmegaConf is unavailable; skipping Torch safe-global registration."
        )
    else:
        safe_globals.extend([DictConfig, ListConfig])

    try:
        from torch.torch_version import TorchVersion
    except ImportError:
        logger.debug(
            "TorchVersion is unavailable; skipping Torch safe-global registration."
        )
    else:
        safe_globals.append(TorchVersion)

    try:
        from pyannote.audio.core.task import Problem, Resolution, Specifications
    except ImportError:
        logger.debug(
            "Pyannote task classes are unavailable; skipping Torch safe-global registration."
        )
    else:
        safe_globals.extend([Problem, Resolution, Specifications])

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


class WhisperTranscriber:
    """WhisperTranscriber handles audio transcription and speaker diarization using openai-whisper.

    Attributes:
        src_lang (str): Source language of the audio.
        task (str): Task type, "transcribe" or "translate".
        n_speakers (int): Maximum number of speakers for diarization.
        start_column (str): DataFrame column for segment start times.
        end_column (str): DataFrame column for segment end times.
        speaker_column (str): DataFrame column for speaker labels.
        text_column (str): DataFrame column for transcribed text.
        transcription_device (str): Device used for model inference ("cuda" or "cpu").
        transcribe_model: Loaded Whisper transcription model.
        diarize_model: Loaded pyannote diarization pipeline.
        audio (np.ndarray): Loaded audio array at 16 kHz.
        transcription_result (Optional[dict[str, Any]]): Result of transcription.
        df (Optional[pd.DataFrame]): DataFrame with transcription results.

    Methods:
        _load_audio(file): Load audio file as numpy array.
        _detect_language(duration_sec, sample_rate): Detect spoken language and load large-v3-turbo.
        _load_transcription_model(preloaded_model): Reuse or swap to the task-specific Whisper model.
        _load_diarization_model(auth_token): Load pyannote diarization pipeline.
        transcription(): Run transcription on the loaded audio file.
        _assign_speakers(diarization_result): Assign speakers from diarization to segments.
        diarization(): Perform speaker diarization and assign labels to segments.
        transcript_output(): Return the transcription result as a DataFrame.
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
        auth_token = HF_HUB_TOKEN

        self.transcription_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.audio = self._load_audio(file_path)
        _configure_torch_safe_globals()

        # Detect language with the transcribe model so it can be reused when the
        # resolved task is "transcribe" (avoids loading the model twice).
        whisper_languages = load_mappings(whisper_language_file)
        det_lang, detect_model = self._detect_language()
        det_lang = det_lang or "en"
        self.src_lang = src_lang if src_lang in whisper_languages.keys() else det_lang
        self.task = "transcribe" if det_lang == trg_lang else task
        logger.info("Using language '{}' for task '{}'", self.src_lang, self.task)

        # Load the transcription model (reusing detect_model when compatible).
        self.transcribe_model = self._load_transcription_model(detect_model)
        self.diarize_model: DiarizationPipeline | None = None
        if self.n_speakers > 1 and not auth_token:
            logger.warning(
                "No Hugging Face token provided. Speaker diarization will be unavailable."
            )
        elif self.n_speakers > 1:
            self.diarize_model = self._load_diarization_model(auth_token)

        # Initialize attributes for transcription results and DataFrame
        self.transcription_result: Optional[dict[str, Any]] = None
        self.df: pd.DataFrame | None = None

    @staticmethod
    def _load_audio(file: Path, sample_rate: int = 16000) -> np.ndarray:
        """Load the audio file as a numpy array using the whisper library.

        Args:
            file (Path): The path to the audio file.
            sample_rate (int): Unused; whisper always resamples to 16000 Hz.

        Returns:
            np.ndarray: The loaded audio as a float32 array at 16 kHz.
        """
        return whisper.load_audio(str(file))

    def _detect_language(
        self, duration_sec: float = 30.0, sample_rate: int = 16000
    ) -> tuple[str | None, Any]:
        """Detect the spoken language and return the loaded Whisper model.

        Loads ``large-v3-turbo`` once and uses it for mel-spectrogram-based
        language detection on the first ``duration_sec`` seconds of audio. The
        loaded model is returned so the caller can reuse it for the transcribe
        task without paying a second load.

        Args:
            duration_sec (float): Number of seconds from the start of the audio to use for detection.
            sample_rate (int): Sample rate for the audio processing. Defaults to 16000.

        Returns:
            tuple[str | None, Any]: Detected language code (e.g., ``"en"``) and
            the loaded Whisper model.

        Raises:
            RuntimeError: If language detection fails.
        """
        sample_frames = int(duration_sec * sample_rate)
        detection_model_id = LOCAL_WHISPER_MODELS["transcribe"]
        logger.info(
            "Loading Whisper model '{}' for language detection...", detection_model_id
        )
        model = whisper.load_model(detection_model_id, device=self.transcription_device)
        audio_clip = whisper.pad_or_trim(self.audio[:sample_frames])
        mel = whisper.log_mel_spectrogram(audio_clip, n_mels=model.dims.n_mels).to(
            model.device
        )
        _, probs = model.detect_language(mel)
        detected_lang = max(probs, key=probs.get)

        if not detected_lang:
            raise RuntimeError("Language detection failed.")

        logger.info("Detected language: {}", detected_lang)
        return detected_lang, model

    def _load_transcription_model(self, preloaded_model: Any) -> Any:
        """Resolve the Whisper model used for the chosen task.

        Reuses ``preloaded_model`` (the ``large-v3-turbo`` instance already
        loaded for language detection) when ``self.task == "transcribe"``. For
        the translate task the preloaded model is released and ``large-v3`` is
        loaded in its place.

        Args:
            preloaded_model (Any): The Whisper model returned by
                :meth:`_detect_language`.

        Returns:
            Any: The Whisper model to use for transcription.
        """
        if self.task == "transcribe":
            logger.info(
                "Reusing Whisper model '{}' for task '{}'.",
                LOCAL_WHISPER_MODELS["transcribe"],
                self.task,
            )
            return preloaded_model

        del preloaded_model
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        translate_model_id = LOCAL_WHISPER_MODELS["translate"]
        logger.info(
            "Loading Whisper model '{}' for task '{}'...",
            translate_model_id,
            self.task,
        )
        return whisper.load_model(translate_model_id, device=self.transcription_device)

    def _load_diarization_model(self, auth_token: str) -> DiarizationPipeline:
        """Load the pyannote diarization pipeline.

        Args:
            auth_token (str): The Hugging Face token for accessing the diarization model.

        Returns:
            DiarizationPipeline: The loaded pyannote speaker diarization pipeline.
        """
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        pipeline = DiarizationPipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=auth_token,
        )
        return pipeline.to(torch.device(device))

    def transcription(self) -> None:
        """Run the transcription process on the loaded audio file using openai-whisper."""
        try:
            result = self.transcribe_model.transcribe(
                self.audio, task=self.task, language=self.src_lang
            )
            self.transcription_result = {"segments": result["segments"]}
        except Exception as e:
            logger.error("Error during transcription: {}", e, exc_info=True)
            raise
        finally:
            del self.transcribe_model
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("Flushed GPU memory.")
            gc.collect()
            logger.info("Resources cleaned up.")

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
        try:
            if self.n_speakers <= 1:
                logger.info("Skipping diarization as only one speaker is specified.")
                return
            if self.diarize_model is None:
                raise RuntimeError(
                    "Speaker diarization requires a valid HF_HUB_TOKEN and "
                    "accepted pyannote model access."
                )

            waveform = torch.tensor(self.audio).unsqueeze(0)
            diarize_result = self.diarize_model(
                {"waveform": waveform, "sample_rate": 16000},
                max_speakers=self.n_speakers,
            )
            self._assign_speakers(diarize_result)
        except Exception as e:
            logger.error("Error during diarization: {}", e, exc_info=True)
            raise
        finally:
            if self.diarize_model is not None:
                del self.diarize_model
                self.diarize_model = None
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("Flushed GPU memory.")
            gc.collect()
            logger.info("Resources cleaned up.")

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
            from openai import OpenAI

            client_kwargs: dict[str, Any] = {"api_key": os.getenv("OPENAI_API_KEY", "")}
            base_url = os.getenv("OPENAI_API_BASE", "").rstrip("/")
            if base_url:
                client_kwargs["base_url"] = base_url
            self._client = OpenAI(**client_kwargs)
        return self._client

    def transcription(self) -> None:
        """Call the external Whisper API and store the segment results.

        Whisper's built-in ``translate`` task always targets English and is
        exposed by OpenAI-compatible servers as a separate ``/v1/audio/translations``
        endpoint. ``task`` itself is not accepted on either endpoint.
        """
        from openai import APIStatusError

        from nextext.utils.env_cfg import load_inference_env

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
        segments = []
        for seg in response.segments:
            segments.append(
                {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                }
            )
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
