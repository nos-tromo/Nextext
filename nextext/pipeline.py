"""Shared pipeline entry points for Nextext processing stages."""

from pathlib import Path
from typing import Any

import pandas as pd
from matplotlib.figure import Figure

from nextext.core.diarization import assign_speakers_by_overlap, diarize_file
from nextext.core.hate_speech import HateSpeechDetector
from nextext.core.ner import extract_entities
from nextext.core.openai_cfg import InferencePipeline
from nextext.core.translation import Translator
from nextext.core.words import WordCounter
from nextext.utils.env_cfg import load_transcription_env

WhisperTranscriber: Any = None
ExternalWhisperTranscriber: Any = None

try:
    from nextext.core.transcription import (
        ExternalWhisperTranscriber as _ExternalWhisperTranscriber,
    )
    from nextext.core.transcription import (
        WhisperTranscriber as _WhisperTranscriber,
    )
except Exception:  # pragma: no cover - environment-specific optional dependency failure
    pass
else:
    WhisperTranscriber = _WhisperTranscriber
    ExternalWhisperTranscriber = _ExternalWhisperTranscriber


def transcription_pipeline(
    file_path: Path,
    src_lang: str,
    n_speakers: int,
) -> tuple[pd.DataFrame, str]:
    """Transcribe the audio file using either a local Whisper model or an external API.

    The transcription provider is derived from ``INFERENCE_PROVIDER``: ``ollama``
    runs the bundled ``openai-whisper`` locally, while ``openai`` and ``vllm``
    forward the audio to an OpenAI-compatible ``/v1/audio/transcriptions``
    endpoint. Whisper always transcribes in the source language; translation to
    a target language is handled separately by :func:`translation_pipeline`.

    Diarization is provider-independent: when ``n_speakers > 1`` the audio is
    sent to the out-of-process ``/diarize`` service (see
    :func:`nextext.core.diarization.diarize_file`) and the returned speaker
    turns are aligned onto the transcript by maximum overlap. It is skipped for
    single-speaker requests and for empty transcripts, and degrades to an
    unlabelled transcript when ``DIARIZE_API_BASE`` is unset or the service is
    unreachable.

    Args:
        file_path (Path): Path to the audio file.
        src_lang (str): Source language code.
        n_speakers (int): Maximum speaker count for diarization. Values greater
            than 1 trigger a ``/diarize`` request; 1 disables diarization.

    Returns:
        tuple[pd.DataFrame, str]: The transcript DataFrame and the
            resolved source language code.
    """
    config = load_transcription_env()

    transcriber: Any
    if config.provider == "external":
        if ExternalWhisperTranscriber is None:
            raise RuntimeError(
                "Transcription dependencies could not be imported. Please verify the openai package installation."
            )
        transcriber = ExternalWhisperTranscriber(
            file_path=file_path,
            src_lang=src_lang,
            model_id=config.whisper_model,
        )
    else:
        if WhisperTranscriber is None:
            raise RuntimeError(
                "Transcription dependencies could not be imported. Verify the openai-whisper "
                "and torchaudio installation."
            )
        transcriber = WhisperTranscriber(
            file_path=file_path,
            src_lang=src_lang,
            n_speakers=n_speakers,
        )

    transcriber.transcription()

    # Diarization runs against the /diarize HTTP service for every provider.
    # Skip it for single-speaker requests and for empty transcripts (silent
    # audio short-circuited by the speech guard) to avoid a needless request.
    segments = transcriber.transcription_result["segments"] if transcriber.transcription_result else []
    if n_speakers > 1 and segments:
        diarize_segments = diarize_file(file_path, max_speakers=n_speakers)
        if diarize_segments:
            assign_speakers_by_overlap(segments, diarize_segments)

    df = transcriber.transcript_output()
    updated_src_lang = transcriber.src_lang or src_lang
    return df, updated_src_lang


def normalize_language_code(lang_code: str | None) -> str | None:
    """Collapse a locale/script code to its base language code.

    Args:
        lang_code (str | None): The language code to normalize, e.g. "en-US" or "de-CH".

    Returns:
        str | None: The base language code, e.g. "en" or "de", or None if the input was None.
    """
    if lang_code is None:
        return None
    return lang_code.split("-", 1)[0]


def should_translate(task: str, src_lang: str | None, trg_lang: str) -> bool:
    """Decide whether the LLM translation stage should run.

    Whisper only transcribes, so every translation is performed downstream by
    the LLM (``TEXT_MODEL``). Translation runs when a ``translate`` task was
    requested and the resolved source language differs from the target: a
    same-language request is a no-op, and an English target is translated like
    any other (there is no longer a Whisper audio-translate hop to English).

    Args:
        task (str): The requested task, ``"transcribe"`` or ``"translate"``.
        src_lang (str | None): The resolved source language code.
        trg_lang (str): The target language code.

    Returns:
        bool: ``True`` when the translation stage should run.
    """
    if task != "translate":
        return False
    return normalize_language_code(src_lang) != normalize_language_code(trg_lang)


def translation_pipeline(
    df: pd.DataFrame,
    trg_lang: str,
    src_lang: str | None = None,
    inference_pipeline: InferencePipeline | None = None,
) -> pd.DataFrame:
    """Translate the transcribed text using a machine translation model.

    Translation is performed only if the target language is different from
    the detected source language.

    Args:
        df (pd.DataFrame): DataFrame containing the transcribed text.
        trg_lang (str): Target language code for translation.
        src_lang (str | None): Source language code, if already known.
        inference_pipeline (InferencePipeline | None): Shared inference client.

    Returns:
        pd.DataFrame: DataFrame with the translated text.
    """
    translator = Translator(inference_pipeline=inference_pipeline)
    resolved_src_lang = src_lang
    if resolved_src_lang is None:
        detected_lang = translator.detect_language(" ".join(df["text"].astype(str).tolist()))
        resolved_src_lang = detected_lang.get("code")
    if normalize_language_code(resolved_src_lang) == normalize_language_code(trg_lang):
        return df
    df["text"] = df["text"].apply(lambda text: translator.translate(trg_lang, text, src_lang=resolved_src_lang))
    return df


def summarization_pipeline(
    text: str,
    inference_pipeline: InferencePipeline,
) -> str:
    """Summarize the given text using a language model and translate the result.

    Args:
        text (str): The text to summarize.
        inference_pipeline (InferencePipeline): An inference pipeline for language model interactions.

    Returns:
        str: The summarized text or None if an error occurs.

    Raises:
        ValueError: If the input text is empty.
    """
    if not text:
        raise ValueError("Text cannot be empty.")
    prompt = inference_pipeline.load_prompt("summary").format(text=text)
    return inference_pipeline.call_model(prompt=prompt)


def wordlevel_pipeline(
    data: pd.DataFrame,
    language: str,
) -> tuple[pd.DataFrame, pd.DataFrame, Figure | None]:
    """Calculate word statistics, named entities, and create a word cloud.

    Args:
        data (pd.DataFrame): DataFrame containing the text data to analyze.
        language (str): Language code of the text data.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, Figure | None]: A tuple containing
            the word-counts DataFrame, the named-entities DataFrame, and the
            word-cloud figure (or ``None`` when there are no word counts to
            plot).
    """
    word_analysis = WordCounter(
        text=" ".join(data["text"].astype(str).tolist()),
        language=language,
    )

    word_analysis.text_to_doc()
    word_analysis.lemmatize_doc()
    word_counts = word_analysis.count_words()
    named_entities = extract_entities(word_analysis.text)
    wordcloud = word_analysis.create_wordcloud()

    return word_counts, named_entities, wordcloud


def hate_speech_pipeline(
    df: pd.DataFrame,
    inference_pipeline: InferencePipeline,
    max_chars: int = 2048,
) -> list[dict[str, Any]]:
    """Detect hate speech in each transcript segment using an LLM.

    Only segments flagged as hate speech are included in the returned list.
    Each entry in the list is a :class:`HateSpeechDetection` dict extended with
    the original ``text`` field for display purposes.

    Args:
        df (pd.DataFrame): Transcript DataFrame with a ``text`` column.
        inference_pipeline (InferencePipeline): Shared inference client.
        max_chars (int): Maximum characters per segment sent for detection. Defaults to 2048.

    Returns:
        list[dict]: Flagged segments, each containing hate_speech, category,
            confidence, reason, text, and start (segment timestamp when available).
    """
    detector = HateSpeechDetector(inference_pipeline, max_chars)
    has_start = "start" in df.columns
    results: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        detection = detector.detect(str(row["text"]))
        if detection["hate_speech"]:
            entry = dict(detection)
            entry["text"] = str(row["text"])
            entry["start"] = str(row["start"]) if has_start else ""
            results.append(entry)
    return results
