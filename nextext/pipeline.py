import getpass
import os
from pathlib import Path

import pandas as pd
from dotenv import find_dotenv, load_dotenv, set_key
from matplotlib.figure import Figure

from nextext.modules.inference_prov_cfg import InferencePipeline
from nextext.modules.translation import Translator
from nextext.modules.words import WordCounter

WhisperTranscriber = None

try:
    from nextext.modules.transcription import WhisperTranscriber as _WhisperTranscriber
except Exception:  # pragma: no cover - environment-specific optional dependency failure
    pass
else:
    WhisperTranscriber = _WhisperTranscriber


def get_api_key(token: str = "API_KEY") -> str:
    """
    Retrieve the API key from environment or prompt user. Works in both local and Docker environments.

    Args:
        token (str): The environment variable to read the API key from.

    Returns:
        str: The API key.

    Raises:
        RuntimeError: If the key cannot be retrieved in a non-interactive (e.g. Docker) environment.
        ValueError: If the key is not found in the environment variables.
    """
    dotenv_path = find_dotenv()
    if dotenv_path:
        load_dotenv(dotenv_path)

    api_key = os.getenv(token)
    if api_key:
        return api_key

    if Path("/.dockerenv").exists():
        raise RuntimeError(
            f"Missing API key. Please set {token} in your .env file or docker-compose environment."
        )

    try:
        api_key = getpass.getpass("Token not found. Please enter your API key: ")
        if dotenv_path:
            set_key(dotenv_path, token, api_key)
        else:
            with open(".env", "w") as env_file:
                env_file.write(f"{token}={api_key}\n")
        return api_key
    except EOFError:
        raise RuntimeError(
            f"API key prompt failed and environment variable {token} is not set."
        )


def transcription_pipeline(
    file_path: Path,
    api_key: str,
    trg_lang: str,
    src_lang: str,
    model_id: str,
    task: str,
    n_speakers: int,
) -> tuple[pd.DataFrame, str]:
    """
    Transcribe and diarize the audio file using WhisperX.

    Args:
        file_path (Path): Path to the audio file.
        api_key (str): API key for authentication.
        trg_lang (str): Target language code for translation check.
        src_lang (str): Source language code.
        model_id (str): Model ID for WhisperX.
        task (str): Task to perform (transcribe or translate).
        n_speakers (int): Number of speakers for diarization.

    Returns:
        pd.DataFrame: DataFrame containing the transcribed text and speaker diarization.
        str: Detected source language code.
    """
    if WhisperTranscriber is None:
        raise RuntimeError(
            "Transcription dependencies could not be imported. Please verify the WhisperX and torchaudio installation."
        )
    transcriber = WhisperTranscriber(
        file_path=file_path,
        auth_token=api_key,
        trg_lang=trg_lang,
        src_lang=src_lang,
        model_id=model_id,
        task=task,
        n_speakers=n_speakers,
    )
    transcriber.transcription()
    if n_speakers > 1:
        transcriber.diarization()
    df = transcriber.transcript_output()
    updated_src_lang = transcriber.src_lang
    updated_src_lang = updated_src_lang if updated_src_lang else src_lang
    return df, updated_src_lang


def translation_pipeline(
    df: pd.DataFrame,
    trg_lang: str,
    src_lang: str | None = None,
    inference_pipeline: InferencePipeline | None = None,
) -> pd.DataFrame:
    """
    Translate the transcribed text using a machine translation model. Translation is performed
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
    if resolved_src_lang == trg_lang:
        return df
    df["text"] = df["text"].apply(
        lambda text: translator.translate(trg_lang, text, src_lang=resolved_src_lang)
    )
    return df


def summarization_pipeline(
    text: str,
    inference_pipeline: InferencePipeline,
) -> str:
    """
    Summarize the given text using a language model and translate the result.

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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, Figure]:
    """
    Calculates word statistics, generates a histogram, and creates a word cloud from the provided text data.

    Args:
        data (pd.DataFrame): DataFrame containing the text data to analyze.
        language (str): Language code of the text data.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: DataFrame with word counts.
            - pd.DataFrame: DataFrame with named entities.
            - pd.DataFrame: DataFrame with noun sentiment.
            - str: Path to the interactive noun graph HTML file.
            - Figure: Word cloud figure.
    """
    word_analysis = WordCounter(
        text=" ".join(data["text"].astype(str).tolist()),
        language=language,
    )

    word_analysis.text_to_doc()
    word_analysis.lemmatize_doc()
    word_counts = word_analysis.count_words()
    named_entities = word_analysis.named_entity_recognition()
    noun_sentiment = word_analysis.get_noun_sentiment()
    noun_graph = word_analysis.create_interactive_graph()
    wordcloud = word_analysis.create_wordcloud()

    return word_counts, named_entities, noun_sentiment, noun_graph, wordcloud
