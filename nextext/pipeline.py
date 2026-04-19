"""Shared pipeline entry points for Nextext processing stages."""

from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
from matplotlib.figure import Figure

from nextext.core.hate_speech import HateSpeechDetector
from nextext.core.openai_cfg import InferencePipeline
from nextext.core.translation import Translator
from nextext.core.words import WordCounter
from nextext.utils.env_cfg import load_transcription_env

WhisperTranscriber: Any = None
ExternalWhisperTranscriber: Any = None

try:
    from nextext.core.transcription import (
        WhisperTranscriber as _WhisperTranscriber,
        ExternalWhisperTranscriber as _ExternalWhisperTranscriber,
    )
except Exception:  # pragma: no cover - environment-specific optional dependency failure
    pass
else:
    WhisperTranscriber = _WhisperTranscriber
    ExternalWhisperTranscriber = _ExternalWhisperTranscriber


def transcription_pipeline(
    file_path: Path,
    trg_lang: str,
    src_lang: str,
    task: str,
    n_speakers: int,
) -> tuple[pd.DataFrame, str]:
    """Transcribe the audio file using either a local Whisper model or an external API.

    The transcription provider is derived from ``INFERENCE_PROVIDER``: ``ollama``
    runs the bundled ``openai-whisper`` locally, while ``openai`` and ``vllm``
    forward the audio to an OpenAI-compatible ``/v1/audio/transcriptions``
    endpoint. External transcription does not support diarization.

    Args:
        file_path (Path): Path to the audio file.
        trg_lang (str): Target language code for translation check.
        src_lang (str): Source language code.
        task (str): Task to perform (transcribe or translate).
        n_speakers (int): Number of speakers for diarization (local provider only).

    Returns:
        tuple[pd.DataFrame, str]: The transcript DataFrame and the
            resolved source language code.
    """
    config = load_transcription_env()

    if config.provider == "external":
        if ExternalWhisperTranscriber is None:
            raise RuntimeError(
                "Transcription dependencies could not be imported. Please verify the openai package installation."
            )
        external_transcriber = ExternalWhisperTranscriber(
            file_path=file_path,
            trg_lang=trg_lang,
            src_lang=src_lang,
            model_id=config.whisper_model,
            task=task,
        )
        external_transcriber.transcription()
        df = external_transcriber.transcript_output()
        updated_src_lang = external_transcriber.src_lang or src_lang
        return df, updated_src_lang
    else:
        if WhisperTranscriber is None:
            raise RuntimeError(
                "Transcription dependencies could not be imported. Please verify the openai-whisper and torchaudio installation."
            )
        local_transcriber = WhisperTranscriber(
            file_path=file_path,
            trg_lang=trg_lang,
            src_lang=src_lang,
            task=task,
            n_speakers=n_speakers,
        )
        local_transcriber.transcription()
        if n_speakers > 1:
            local_transcriber.diarization()
        df = local_transcriber.transcript_output()
        updated_src_lang = local_transcriber.src_lang or src_lang
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


def translation_pipeline(
    df: pd.DataFrame,
    trg_lang: str,
    src_lang: str | None = None,
    inference_pipeline: InferencePipeline | None = None,
) -> pd.DataFrame:
    """Translate the transcribed text using a machine translation model. Translation is performed
    only if the target language is different from the detected source language.

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
        detected_lang = translator.detect_language(
            " ".join(df["text"].astype(str).tolist())
        )
        resolved_src_lang = detected_lang.get("code")
    if normalize_language_code(resolved_src_lang) == normalize_language_code(trg_lang):
        return df
    df["text"] = df["text"].apply(
        lambda text: translator.translate(trg_lang, text, src_lang=resolved_src_lang)
    )
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
    """Calculates word statistics, named entities, and creates a word cloud from the provided text data.

    Args:
        data (pd.DataFrame): DataFrame containing the text data to analyze.
        language (str): Language code of the text data.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: DataFrame with word counts.
            - pd.DataFrame: DataFrame with named entities.
            - Figure | None: Word cloud figure, or ``None`` when there are no
              word counts to plot.
    """
    word_analysis = WordCounter(
        text=" ".join(data["text"].astype(str).tolist()),
        language=language,
    )

    word_analysis.text_to_doc()
    word_analysis.lemmatize_doc()
    word_counts = word_analysis.count_words()
    named_entities = word_analysis.named_entity_recognition()
    wordcloud = word_analysis.create_wordcloud()

    return word_counts, named_entities, wordcloud


def hate_speech_pipeline(
    df: pd.DataFrame,
    inference_pipeline: InferencePipeline,
    max_chars: int = 2048,
) -> list[dict]:
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
            confidence, reason, and text.
    """
    detector = HateSpeechDetector(inference_pipeline, max_chars)
    results: list[dict] = []
    for _, row in df.iterrows():
        detection = detector.detect(str(row["text"]))
        if detection["hate_speech"]:
            entry = dict(detection)
            entry["text"] = str(row["text"])
            results.append(entry)
    return results
