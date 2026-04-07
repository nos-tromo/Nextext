"""Audio transcription and diarization with WhisperX."""

import gc
import os
import warnings
from datetime import timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import whisperx
from dotenv import load_dotenv
from loguru import logger
from whisperx.diarize import DiarizationPipeline

from nextext.utils.mappings_loader import load_mappings

# Suppress NNPACK warnings emitted by PyTorch when the CPU lacks required
# SIMD support (common inside Docker containers).  Without this filter the
# Silero VAD model floods stderr with thousands of identical warnings.
warnings.filterwarnings("ignore", message=".*Could not initialize NNPACK.*")
load_dotenv()
HF_HUB_TOKEN = os.getenv("HF_HUB_TOKEN", "")
TRANSCRIPTION_VAD_METHOD = "silero"


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


class WhisperTranscriber:
    """WhisperTranscriber handles audio transcription and speaker diarization using WhisperX.

    Attributes:
        src_lang (str): Source language of the audio.
        task (str): Task type, "transcribe" or "translate".
        n_speakers (int): Maximum number of speakers for diarization.
        start_column (str): DataFrame column for segment start times.
        end_column (str): DataFrame column for segment end times.
        speaker_column (str): DataFrame column for speaker labels.
        text_column (str): DataFrame column for transcribed text.
        transcription_device (str): Device used for model inference ("cuda", "cpu", or "mps").
        transcribe_model: Loaded WhisperX transcription model.
        model_a: Loaded WhisperX alignment model.
        metadata (dict): Metadata for the alignment model.
        diarize_model: Loaded WhisperX diarization model.
        audio (torch.Tensor): Loaded audio tensor.
        transcription_result (Optional[dict[str, Any]]): Result of transcription and alignment.
        df (Optional[pd.DataFrame]): DataFrame with diarization/transcription results.

    Methods:
        _load_audio(file): Load audio file as tensor.
        _detect_language(duration_sec, sample_rate): Detect spoken language in the audio.
        _load_transcription_model(model_id): Load WhisperX model for transcription and alignment.
        _load_diarization_model(auth_token): Load WhisperX diarization pipeline.
        transcription(batch_size): Run transcription and alignment.
        _seconds_to_time(seconds): Convert seconds to HH:MM:SS string.
        diarization(n_speakers): Perform speaker diarization and return DataFrame.
    """

    def __init__(
        self,
        file_path: Path,
        trg_lang: str,
        src_lang: str | None = None,
        model_id: str = "default",
        task: str = "transcribe",
        n_speakers: int = 1,
        start_column: str = "start",
        end_column: str = "end",
        speaker_column: str = "speaker",
        text_column: str = "text",
        whisper_language_file: str = "whisper_languages.json",
        whisper_model_file: str = "whisper_models.json",
    ) -> None:
        """Initialize the WhisperTranscriber object.

        Args:
            file_path (str): The path to the input file.
            trg_lang (str): The target language for translation.
            src_lang (str, optional): The source language of the file. Defaults to None, which triggers language detection.
            model_id (str, optional): The size of the Whisper model. Defaults to "default".
            task (str): Indicates whether the task is transcription or translation. Defaults to "transcribe".
            n_speakers (int): The maximum number of speakers to identify in the audio. Defaults to 1.
            start_column (str): The text column with the starting timestamp. Defaults to "start".
            end_column(str): The text column with the ending timestamp. Defaults to "end".
            speaker_column (str): The text column with the speaker information. Defaults to "speaker".
            text_column (str): The text column where the result is stored. Defaults to "text".
            whisper_language_file (str): Path to the Whisper language mapping file.
            whisper_model_file (str): Path to the Whisper model configuration file.
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

        # Select or detect language
        whisper_languages = load_mappings(whisper_language_file)
        det_lang = self._detect_language() or "en"
        self.src_lang = src_lang if src_lang in whisper_languages.keys() else det_lang
        self.task = "transcribe" if det_lang == trg_lang else task
        logger.info("Using language '{}' for task '{}'", self.src_lang, self.task)

        # Load the transcription and alignment models
        self.transcribe_model, self.align_model, self.align_metadata = (
            self._load_transcription_model(model_id, whisper_model_file)
        )
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
        """Load the audio file as a tensor using the WhisperX library.

        Args:
            file (Path): The path to the audio file.
            sample_rate (int): Sample rate for the audio processing. Defaults to 16000.

        Returns:
            np.ndarray: The loaded audio as a tensor.
        """
        return whisperx.load_audio(file=file, sr=sample_rate)

    def _detect_language(
        self, duration_sec: float = 30.0, sample_rate: int = 16000
    ) -> str | None:
        """Detect the spoken language in the audio file using WhisperX.
        This method processes only the first `duration_sec` seconds of audio.

        Args:
            duration_sec (float): Number of seconds from the start of the audio to use for detection.
            sample_rate (int): Sample rate for the audio processing. Defaults to 16000.

        Returns:
            str: Detected language code (e.g., "en", "de"), or None if detection fails.

        Raises:
            RuntimeError: If language detection fails or the audio is too short.
        """
        sample_frames = int(duration_sec * sample_rate)
        clipped_audio = self.audio[:sample_frames]

        # Load a temporary model instance with language=None to trigger auto-detection
        logger.info("Detecting language using WhisperX...")
        temp_model = whisperx.load_model(
            whisper_arch="small",
            device=self.transcription_device,
            compute_type="float16" if torch.cuda.is_available() else "int8",
            language=None,
            vad_method=TRANSCRIPTION_VAD_METHOD,
        )
        try:
            result = temp_model.transcribe(clipped_audio)
        finally:
            del temp_model
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        detected_lang = result.get("language")
        if not detected_lang:
            raise RuntimeError("Language detection failed.")

        logger.info("Detected language: {}", detected_lang)
        return detected_lang

    def _load_transcription_model(
        self, model_id: str, whisper_model_file: str
    ) -> tuple[Any, Any | None, dict[str, Any] | None]:
        """Load the Whisper model for transcription. If alignment model is unavailable, skip it.

        Args:
            model_id (str): The size of the Whisper model (tiny, base, small, medium, large, turbo).
            whisper_model_file (str): Path to the Whisper model configuration file.

        Returns:
            Any: The loaded Whisper model for transcription.
            Any or None: The loaded Whisper model for alignment, or None if unavailable.
            dict[str, Any] or None: Metadata about the loaded model, or None if unavailable.

        Raises:
            ValueError: If the model configuration is invalid.
        """
        model_config = load_mappings(whisper_model_file)
        config_key = f"{model_id}_{self.task}" if model_id == "default" else model_id
        mapped_model_id = model_config.get(config_key)
        logger.info(
            "Loading Whisper model '{}' for task '{}'...",
            mapped_model_id,
            self.task,
        )

        if mapped_model_id is None:
            raise ValueError(f"Invalid model configuration for key '{config_key}'.")

        dtype = "float16" if torch.cuda.is_available() else "int8"
        transcribe_model = whisperx.load_model(
            whisper_arch=mapped_model_id,
            device=self.transcription_device,
            compute_type=dtype,
            language=None if self.task == "translate" else self.src_lang,
            vad_method=TRANSCRIPTION_VAD_METHOD,
        )
        try:
            align_model, align_metadata = whisperx.load_align_model(
                language_code=self.src_lang, device=self.transcription_device
            )
        except Exception as align_exc:
            logger.warning(
                "Alignment model could not be loaded: {}. Continuing without alignment.",
                align_exc,
            )
            align_model, align_metadata = None, None
        return transcribe_model, align_model, align_metadata

    def _load_diarization_model(self, auth_token: str) -> DiarizationPipeline:
        """Load the WhisperX model for speaker diarization.

        Args:
            auth_token (str): The Hugging Face token for accessing the diarization model.

        Returns:
            DiarizationPipeline: The loaded WhisperX model for speaker diarization.
        """
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        return DiarizationPipeline(use_auth_token=auth_token, device=device)

    def transcription(self, batch_size: int = 16) -> None:
        """Run the transcription process on the loaded audio file using the WhisperX library.

        Args:
            batch_size (int, optional): The number of audio segments to process in each batch during transcription. Defaults to 16.
        """
        try:
            result = self.transcribe_model.transcribe(
                self.audio, batch_size=batch_size, task=self.task
            )
            if self.task == "transcribe" and self.align_model is not None:
                self.transcription_result = whisperx.align(
                    transcript=result["segments"],
                    model=self.align_model,
                    align_model_metadata=self.align_metadata,
                    audio=self.audio,
                    device=self.transcription_device,
                    return_char_alignments=False,
                )
            else:
                logger.info(
                    "Skipping forced alignment for task '{}' and language '{}'.",
                    self.task,
                    self.src_lang,
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

    @staticmethod
    def _seconds_to_time(seconds: float) -> str:
        """Convert seconds to a string representation of time in the format HH:MM:SS.

        Args:
            seconds (float): The number of seconds to convert.

        Returns:
            str: The string representation of time in the format HH:MM:SS.
        """
        return str(timedelta(seconds=round(seconds)))

    def diarization(self) -> None:
        """Perform speaker diarization on the audio file.

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

            diarize_segments = self.diarize_model(
                self.audio, max_speakers=self.n_speakers
            )
            self.transcription_result = whisperx.assign_word_speakers(
                diarize_segments, self.transcription_result
            )
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
        """Check if the given text ends with a sentence-ending punctuation mark.

        Args:
            text (str): The text string to check.

        Returns:
            bool: True if the text ends with a supported sentence-ending punctuation mark, otherwise False.
        """
        return text.strip().endswith((".", "!", "?", "؟", "۔"))

    def _merge_transcriptions_by_sentence(self, data: pd.DataFrame) -> pd.DataFrame:
        """Merge transcriptions by sentences based on punctuation.

        Args:
            data (pd.DataFrame): The original DataFrame containing transcription data.

        Returns:
            pd.DataFrame: A new DataFrame with merged sentences and adjusted timestamps.
        """
        output_columns: list[str] = [
            self.start_column,
            self.end_column,
        ]
        has_speaker = self.speaker_column in data.columns
        if has_speaker:
            output_columns.append(self.speaker_column)
        output_columns.append(self.text_column)
        if data.empty or self.text_column not in data.columns:
            logger.warning("No transcription rows were available for sentence merging.")
            return pd.DataFrame(columns=pd.Index(output_columns))

        def _build_empty_row() -> dict[str, Any]:
            """Create an empty row for sentence aggregation.

            Returns:
                dict[str, Any]: A mutable row container for sentence merging.
            """
            row: dict[str, Any] = {
                self.start_column: None,
                self.end_column: None,
                self.text_column: "",
            }
            if has_speaker:
                row[self.speaker_column] = None
            return row

        def _append_current_row(end_value: Any) -> None:
            """Append the current aggregated row if it contains text.

            Args:
                end_value (Any): The end timestamp to assign before append.
            """
            current_text = str(current_row[self.text_column] or "").strip()
            if not current_text:
                return
            current_row[self.end_column] = end_value
            current_row[self.text_column] = current_text
            new_rows.append(current_row.copy())

        new_rows: list[dict[str, Any]] = []
        current_row = _build_empty_row()
        previous_end: Any = None

        for _, row in data.iterrows():
            row_speaker = row.get(self.speaker_column) if has_speaker else None
            current_text = str(current_row[self.text_column] or "").strip()
            if (
                has_speaker
                and current_text
                and current_row.get(self.speaker_column) != row_speaker
            ):
                _append_current_row(previous_end)
                current_row = _build_empty_row()

            if current_row.get(self.start_column) is None:
                current_row[self.start_column] = row.get(self.start_column)
            if has_speaker and current_row.get(self.speaker_column) is None:
                current_row[self.speaker_column] = row_speaker

            if row[self.text_column]:
                current_row[self.text_column] += row[self.text_column].strip() + " "

            previous_end = row.get(self.end_column)
            if self._ends_with_punctuation(row[self.text_column]):
                _append_current_row(previous_end)
                current_row = _build_empty_row()

        if str(current_row[self.text_column] or "").strip():
            _append_current_row(previous_end)

        merged_df = pd.DataFrame(new_rows, columns=pd.Index(output_columns))
        logger.info("Transcriptions successfully merged by sentence.")
        return merged_df

    def transcript_output(self) -> pd.DataFrame:
        """Get the transcription result as a DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the transcription results.

        Raises:
            ValueError: If the transcription result is not available or if the transcription has not been run yet.
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
                self._seconds_to_time(item["start"]),
                self._seconds_to_time(item["end"]),
            ]
            if has_speaker:
                row.append(item.get("speaker", "Unknown"))
            row.append(item["text"])
            segments.append(row)

        # Build columns list dynamically
        columns: list[str] = [self.start_column, self.end_column]
        if has_speaker:
            columns.append(self.speaker_column)
        columns.append(self.text_column)

        df = pd.DataFrame(segments, columns=pd.Index(columns))
        if self.n_speakers <= 1 and has_speaker:
            df.drop(self.speaker_column, axis=1, inplace=True)
        self.df = self._merge_transcriptions_by_sentence(df)
        return self.df
