"""Shared pipeline entry points for Nextext processing stages."""

from pathlib import Path
from typing import Any

import pandas as pd
from matplotlib.figure import Figure

from nextext.core.hate_speech import HateSpeechDetector
from nextext.core.openai_cfg import InferencePipeline
from nextext.core.transcription import ExternalWhisperTranscriber
from nextext.core.translation import Translator
from nextext.core.words import WordCounter
from nextext.utils.env_cfg import load_whisper_env


def transcription_pipeline(
    file_path: Path,
    trg_lang: str,
    src_lang: str,
    task: str,
    n_speakers: int,
) -> tuple[pd.DataFrame, str]:
    """Transcribe the audio file via the external Whisper API.

    The audio always goes to an OpenAI-compatible
    ``/v1/audio/transcriptions`` endpoint resolved by
    :func:`nextext.utils.env_cfg.load_whisper_env`. Requesting more than one
    speaker additionally calls the external diarization service (see
    :mod:`nextext.core.diarization`).

    Args:
        file_path (Path): Path to the audio file.
        trg_lang (str): Target language code for translation check.
        src_lang (str): Source language code.
        task (str): Task to perform (transcribe or translate).
        n_speakers (int): Maximum number of speakers for diarization.

    Returns:
        tuple[pd.DataFrame, str]: The transcript DataFrame and the
            resolved source language code.
    """
    config = load_whisper_env()
    transcriber = ExternalWhisperTranscriber(
        file_path=file_path,
        trg_lang=trg_lang,
        src_lang=src_lang,
        model_id=config.model,
        task=task,
        n_speakers=n_speakers,
    )
    transcriber.transcription()
    if n_speakers > 1:
        transcriber.diarization()
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
    named_entities = word_analysis.named_entity_recognition()
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
