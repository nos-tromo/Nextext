from __future__ import annotations

import gc
import logging
from datetime import timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import whisperx
from whisperx.diarize import DiarizationPipeline

from nextext.utils import load_mappings


class WhisperTranscriber:
    """
    WhisperTranscriber handles audio transcription and speaker diarization using WhisperX.

    Attributes:
        language (str): Source language of the audio.
        start_column (str): DataFrame column for segment start times.
        end_column (str): DataFrame column for segment end times.
        speaker_column (str): DataFrame column for speaker labels.
        text_column (str): DataFrame column for transcribed text.
        task (str): Task type, "transcribe" or "translate".
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
        language: str,
        auth_token: str,
        model_id: str = "default",
        task: str = "transcribe",
        start_column: str = "start",
        end_column: str = "end",
        speaker_column: str = "speaker",
        text_column: str = "text",
        whisper_language_file: str = "whisper_languages.json",
        whisper_model_file: str = "whisper_models.json",
    ) -> None:
        """
        Initialize the WhisperTranscriber object.

        Args:
            file_path (str): The path to the input file.
            language (str): The source language of the file.
            auth_token (str): The Hugging Face token for accessing the diarization model.
            model_id (str, optional): The size of the Whisper model. Defaults to "default".
            task (str): Indicates whether the task is transcription or translation. Defaults to "transcribe".
            start_column (str): The text column with the starting timestamp. Defaults to "start".
            end_column(str): The text column with the ending timestamp. Defaults to "end".
            speaker_column (str): The text column with the speaker information. Defaults to "speaker".
            text_column (str): The text column where the result is stored. Defaults to "text".
            whisper_language_file (str): Path to the Whisper language mapping file.
            whisper_model_file (str): Path to the Whisper model configuration file.
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        self.start_column = start_column
        self.end_column = end_column
        self.speaker_column = speaker_column
        self.text_column = text_column
        self.task = task

        self.transcription_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.audio = self._load_audio(file_path)

        # Detect language if set to "detect", otherwise use specified language
        whisper_languages = load_mappings(whisper_language_file)
        if language == "detect":
            self.logger.info(
                "Language set to 'detect'; running detection before loading model."
            )
            detected_language = self._detect_language()
            self.language = (
                detected_language
                if detected_language in whisper_languages.keys()
                else "en"
            )
        else:
            self.logger.info(f"Using specified language: {language}")
            self.language = (
                language
                if language in whisper_languages.keys()
                else self._detect_language()
                if self._detect_language() in whisper_languages.keys()
                else "en"
            )
        self.logger.info(f"Using language: {self.language}")

        # Load the transcription and alignment models
        self.transcribe_model, self.model_a, self.metadata = (
            self._load_transcription_model(model_id, whisper_model_file)
        )
        self.diarize_model = self._load_diarization_model(auth_token)

        # Initialize attributes for transcription results and DataFrame
        self.transcription_result: Optional[dict[str, Any]] = None
        self.df: pd.DataFrame | None = None

    @staticmethod
    def _load_audio(file: Path, sample_rate: int = 16000) -> np.ndarray:
        """
        Load the audio file as a tensor using the WhisperX library.

        Args:
            file (Path): The path to the audio file.
            sample_rate (int): Sample rate for the audio processing. Defaults to 16000.

        Returns:
            np.ndarray: The loaded audio as a tensor.
        """
        try:
            return whisperx.load_audio(file=file, sr=sample_rate)
        except Exception as e:
            logging.error(f"Error loading audio file '{file}': {e}", exc_info=True)
            raise

    def _detect_language(
        self, duration_sec: float = 30.0, sample_rate: int = 16000
    ) -> str | None:
        """
        Detect the spoken language in the audio file using WhisperX.
        This method processes only the first `duration_sec` seconds of audio.

        Args:
            duration_sec (float): Number of seconds from the start of the audio to use for detection.
            sample_rate (int): Sample rate for the audio processing. Defaults to 16000.

        Returns:
            str: Detected language code (e.g., "en", "de"), or None if detection fails.
        """
        try:
            sample_frames = int(duration_sec * sample_rate)
            clipped_audio = self.audio[:sample_frames]

            # Load a temporary model instance with language=None to trigger auto-detection
            self.logger.info("Detecting language using WhisperX...")
            temp_model = whisperx.load_model(
                whisper_arch="small",
                device=self.transcription_device,
                compute_type="float16" if torch.cuda.is_available() else "int8",
                language=None,
            )
            result = temp_model.transcribe(clipped_audio)

            detected_lang = result.get("language")
            if not detected_lang:
                raise RuntimeError("Language detection failed.")

            self.logger.info(f"Detected language: {detected_lang}")
            return detected_lang

        except Exception as e:
            self.logger.error(f"Error detecting language: {e}", exc_info=True)
            return None

    def _load_transcription_model(
        self, model_id: str, whisper_model_file: str
    ) -> tuple[Any, Any, dict[str, Any]]:
        """
        Load the Whisper model for transcription.

        Args:
            model_id (str): The size of the Whisper model (tiny, base, small, medium, large, turbo).
            whisper_model_file (str): Path to the Whisper model configuration file.

        Returns:
            Any: The loaded Whisper model for transcription.
            Any: The loaded Whisper model for alignment.
            dict[str, Any]: Metadata about the loaded model.
        """
        try:
            model_config = load_mappings(whisper_model_file)
            config_key = (
                f"{model_id}_{self.task}" if model_id == "default" else model_id
            )
            mapped_model_id = model_config.get(config_key)
            self.logger.info(
                f"Loading Whisper model '{mapped_model_id}' for task '{self.task}'..."
            )

            if mapped_model_id is None:
                raise ValueError(f"Invalid model configuration for key '{config_key}'.")

            dtype = "float16" if torch.cuda.is_available() else "int8"
            model = whisperx.load_model(
                whisper_arch=mapped_model_id,
                device=self.transcription_device,
                compute_type=dtype,
                language=None if self.task == "translate" else self.language,
            )
            model_a, metadata = whisperx.load_align_model(
                language_code=self.language, device=self.transcription_device
            )
            return model, model_a, metadata
        except Exception as e:
            self.logger.error(
                f"Error setting up model '{model_id}': {e}", exc_info=True
            )
            raise

    def _load_diarization_model(self, auth_token: str) -> DiarizationPipeline:
        """
        Load the WhisperX model for speaker diarization.

        Args:
            auth_token (str): The Hugging Face token for accessing the diarization model.

        Returns:
            DiarizationPipeline: The loaded WhisperX model for speaker diarization.
        """
        try:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
            return DiarizationPipeline(use_auth_token=auth_token, device=device)
        except Exception as e:
            self.logger.error(f"Error setting up diarization model: {e}", exc_info=True)
            raise

    def transcription(self, batch_size: int = 16) -> None:
        """
        Run the transcription process on the loaded audio file using the WhisperX library.

        Args:
            batch_size (int, optional): The number of audio segments to process in each batch during transcription. Defaults to 16.
        """
        try:
            result = self.transcribe_model.transcribe(
                self.audio, batch_size=batch_size, task=self.task
            )
            if self.task == "transcribe":
                self.transcription_result = whisperx.align(
                    transcript=result["segments"],
                    model=self.model_a,
                    align_model_metadata=self.metadata,
                    audio=self.audio,
                    device=self.transcription_device,
                    return_char_alignments=False,
                )
            else:
                self.transcription_result = {"segments": result["segments"]}
        except Exception as e:
            self.logger.error(f"Error during transcription: {e}", exc_info=True)
            raise
        finally:
            del self.transcribe_model
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.info("Flushed GPU memory.")
            gc.collect()
            self.logger.info("Resources cleaned up.")

    @staticmethod
    def _seconds_to_time(seconds: float) -> str:
        """
        Convert seconds to a string representation of time in the format HH:MM:SS.

        Args:
            seconds (float): The number of seconds to convert.

        Returns:
            str: The string representation of time in the format HH:MM:SS.
        """
        try:
            return str(timedelta(seconds=round(seconds)))
        except Exception as e:
            logging.error(f"Error converting seconds to time: {e}", exc_info=True)
            raise

    def diarization(self, n_speakers: int = 1) -> pd.DataFrame:
        """
        Perform speaker diarization on the audio file.

        Args:
            num_speakers (int): The number of speakers to identify in the audio. Defaults to 1.

        Returns:
            pd.DataFrame: A DataFrame containing the diarization results with start and end times, speaker labels, and text.
        """
        try:
            diarize_segments = self.diarize_model(self.audio, max_speakers=n_speakers)
            combined = whisperx.assign_word_speakers(
                diarize_segments, self.transcription_result
            )
            segments = [
                (
                    self._seconds_to_time(item["start"]),
                    self._seconds_to_time(item["end"]),
                    item.get("speaker", "Unknown"),
                    item["text"],
                )
                for item in combined["segments"]
            ]
            self.df = pd.DataFrame(
                segments,
                columns=[
                    self.start_column,
                    self.end_column,
                    self.speaker_column,
                    self.text_column,
                ],
            )
            return self.df
        except Exception as e:
            self.logger.error(f"Error during diarization: {e}", exc_info=True)
            raise
        finally:
            del self.diarize_model
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.info("Flushed GPU memory.")
            gc.collect()
            self.logger.info("Resources cleaned up.")
