"""Audio transcription via an external OpenAI-compatible Whisper API.

An external ``/vad`` speech guard (see :func:`nextext.core.vad.has_speech`)
keeps silent and noise-only audio away from the remote endpoint; speaker
diarization is delegated to the external service behind
:func:`nextext.core.diarization.diarize_file`.
"""

from datetime import timedelta
from pathlib import Path as Path  # re-exported for tests that monkeypatch nextext.core.transcription.Path
from typing import Any

import pandas as pd
from loguru import logger
from openai import APIStatusError, OpenAI

from nextext.core.diarization import diarize_file
from nextext.core.vad import has_speech
from nextext.utils.env_cfg import load_inference_env, load_whisper_env
from nextext.utils.mappings_loader import load_mappings

OPENAI_WHISPER_MAX_UPLOAD_BYTES: int = 25 * 1024 * 1024

# Segments with ``no_speech_prob`` above this threshold are discarded.
# Whisper's built-in filter requires *both* high ``no_speech_prob`` and low
# ``avg_logprob``, which lets confident hallucinations on silent audio slip
# through.  Filtering on ``no_speech_prob`` alone catches them.
NO_SPEECH_THRESHOLD: float = 0.6


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


def _assign_speakers(
    segments: list[dict[str, Any]],
    speaker_turns: list[dict[str, Any]],
) -> None:
    """Assign speaker labels to transcription segments by maximum overlap.

    Args:
        segments (list[dict[str, Any]]): Whisper segment dicts carrying
            ``start``/``end`` seconds; mutated in place — the winning
            ``speaker`` label is written into each overlapping segment.
        speaker_turns (list[dict[str, Any]]): Diarization speaker turns as
            ``{"start", "end", "speaker"}`` dicts, as returned by
            :func:`nextext.core.diarization.diarize_file`.
    """
    for segment in segments:
        seg_start = float(segment["start"])
        seg_end = float(segment["end"])
        speaker_durations: dict[str, float] = {}
        for turn in speaker_turns:
            overlap_start = max(seg_start, float(turn["start"]))
            overlap_end = min(seg_end, float(turn["end"]))
            if overlap_end > overlap_start:
                speaker = str(turn["speaker"])
                speaker_durations[speaker] = speaker_durations.get(speaker, 0.0) + (overlap_end - overlap_start)
        if speaker_durations:
            segment["speaker"] = max(speaker_durations, key=lambda s: speaker_durations[s])


class ExternalWhisperTranscriber:
    """Transcribe audio via an external OpenAI-compatible Whisper API.

    Speaker diarization is delegated to the external diarization service
    (see :func:`nextext.core.diarization.diarize_file`) and merged into the
    transcribed segments by maximum overlap.

    Attributes:
        file_path (Path): Path to the audio file.
        src_lang (str | None): Source language code; populated from API response if not provided.
        task (str): Task type, "transcribe" or "translate".
        n_speakers (int): Maximum number of speakers for diarization;
            values above 1 trigger the external diarization call.

    Methods:
        transcription(): Call the external API and store segment results.
        diarization(): Label segments with speakers via the external service.
        transcript_output(): Return the transcription result as a DataFrame.
    """

    def __init__(
        self,
        file_path: Path,
        trg_lang: str | None = None,
        src_lang: str | None = None,
        model_id: str = "whisper-1",
        task: str = "transcribe",
        n_speakers: int = 1,
        start_column: str = "start",
        end_column: str = "end",
        speaker_column: str = "speaker",
        text_column: str = "text",
    ) -> None:
        """Initialize the ExternalWhisperTranscriber.

        Args:
            file_path (Path): Path to the audio file.
            trg_lang (str | None): Accepted for interface compatibility; not forwarded to the API.
            src_lang (str | None): Source language code. Defaults to None (API auto-detects).
            model_id (str): Model name to pass to the external API. Defaults to "whisper-1".
            task (str): "transcribe" or "translate". Defaults to "transcribe".
            n_speakers (int): Maximum number of speakers for diarization. Defaults to 1
                (diarization disabled).
            start_column (str): DataFrame column for segment start times.
            end_column (str): DataFrame column for segment end times.
            speaker_column (str): DataFrame column for speaker labels.
            text_column (str): DataFrame column for transcribed text.
        """
        self.file_path = file_path
        self.src_lang = src_lang
        self.task = task
        self._model_id = model_id
        self.n_speakers = n_speakers
        self.start_column = start_column
        self.end_column = end_column
        self.speaker_column = speaker_column
        self.text_column = text_column
        self.transcription_result: dict[str, Any] | None = None
        self._client: Any = None

    @property
    def _get_client(self) -> Any:
        """Lazily create the OpenAI-compatible client from the Whisper endpoint config.

        The endpoint resolves via
        :func:`nextext.utils.env_cfg.load_whisper_env`: the dedicated
        ``WHISPER_API_BASE``/``WHISPER_API_KEY`` pair when set, else the
        central ``OPENAI_API_BASE``/``OPENAI_API_KEY``.

        Returns:
            Any: The cached OpenAI client instance.
        """
        if self._client is None:
            cfg = load_whisper_env()
            client_kwargs: dict[str, Any] = {"api_key": cfg.api_key}
            base_url = cfg.api_base.rstrip("/")
            if base_url:
                client_kwargs["base_url"] = base_url
            self._client = OpenAI(**client_kwargs)
        return self._client

    def transcription(self) -> None:
        """Call the external Whisper API and store the segment results.

        Two hallucination guards bracket the request:

        1. **External /vad guard** (see :func:`nextext.core.vad.has_speech`)
           — the file is screened by the ``/vad`` service before any
           Whisper request, so silent / noise-only files never reach the
           paid endpoint. The guard is fail-open: an unset or unreachable
           service lets the file through.
        2. **no_speech_prob post-filter** — segments whose
           ``no_speech_prob`` exceeds :data:`NO_SPEECH_THRESHOLD` are
           dropped from the returned payload.

        Whisper's built-in ``translate`` task always targets English and
        is exposed by OpenAI-compatible servers as a separate
        ``/v1/audio/translations`` endpoint. ``task`` itself is not
        accepted on either endpoint.
        """
        if not has_speech(self.file_path):
            logger.warning(
                "VAD service reported no speech in {}; skipping external transcription request.",
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
                "Compress or split the file before retrying, or point "
                "WHISPER_API_BASE at a self-hosted endpoint without a hard cap "
                "(e.g. the vllm-service audio container)."
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
            self.src_lang = _normalize_whisper_language(getattr(response, "language", None))
        logger.info(
            "External transcription complete: {} segments, language={}",
            len(segments),
            self.src_lang,
        )

    def diarization(self) -> None:
        """Label the transcribed segments with speakers via the external service.

        Uploads the audio file to the diarization endpoint (see
        :func:`nextext.core.diarization.diarize_file`) and assigns each
        transcribed segment the speaker with the maximum temporal overlap.
        Skipped when ``n_speakers <= 1`` or no segments survived
        transcription (nothing to label — the upload would be wasted).

        The diarization client is fail-soft: when ``DIARIZE_API_BASE`` is
        unset or the service is unreachable it returns no speaker turns and
        the segments are left unlabelled (see :mod:`nextext.core.diarization`).

        Raises:
            ValueError: If transcription has not been run yet.
        """
        if self.n_speakers <= 1:
            logger.info("Skipping diarization as only one speaker is specified.")
            return
        if self.transcription_result is None or "segments" not in self.transcription_result:
            raise ValueError("Transcription result is not available. Run transcription first.")
        segments = self.transcription_result["segments"]
        if not segments:
            logger.info("No transcribed segments to diarize; skipping diarization request.")
            return
        speaker_turns = diarize_file(self.file_path, max_speakers=self.n_speakers)
        _assign_speakers(segments, speaker_turns)

    def transcript_output(self) -> pd.DataFrame:
        """Get the external transcription result as a DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the transcription results,
            including a speaker column when diarization labeled segments.

        Raises:
            ValueError: If transcription has not been run yet.
        """
        if self.transcription_result is None or "segments" not in self.transcription_result:
            raise ValueError("Transcription result is not available. Run transcription first.")

        rows = []
        has_speaker = any("speaker" in item for item in self.transcription_result["segments"])
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
        if self.n_speakers <= 1 and has_speaker:
            df.drop(self.speaker_column, axis=1, inplace=True)
        return _merge_transcriptions_by_sentence(
            df,
            self.start_column,
            self.end_column,
            self.speaker_column,
            self.text_column,
        )
