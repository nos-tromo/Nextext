import getpass
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import find_dotenv, load_dotenv, set_key

from nextext.modules import call_ollama, text_summarization_prompt
from nextext.modules import TopicModeling
from nextext.modules import ToxClassifier
from nextext.modules import WhisperTranscriber
from nextext.modules import Translator
from nextext.modules import WordCounter

__all__ = [
    "get_api_key",
    "transcription_pipeline",
    "translation_pipeline",
    "summarization_pipeline",
    "wordlevel_pipeline",
    "topics_pipeline",
    "toxicity_pipeline",
]


def get_api_key(token: str = "API_KEY") -> str:
    """
    Retrieve the API key from environment or prompt user. Works in both local and Docker environments.

    Args:
        token (str): The environment variable to read the API key from.

    Returns:
        str: The API key.

    Raises:
        RuntimeError: If the key cannot be retrieved in a non-interactive (e.g. Docker) environment.
    """
    dotenv_path = find_dotenv()
    if dotenv_path:
        load_dotenv(dotenv_path)

    api_key = os.getenv(token)
    if api_key:
        return api_key

    if Path("/.dockerenv").exists():
        raise RuntimeError(f"Missing API key. Please set {token} in your .env file or docker-compose environment.")

    try:
        api_key = getpass.getpass("Token not found. Please enter your API key: ")
        if dotenv_path:
            set_key(dotenv_path, token, api_key)
        else:
            with open(".env", "w") as env_file:
                env_file.write(f"{token}={api_key}\n")
        return api_key
    except EOFError:
        raise RuntimeError(f"API key prompt failed and environment variable {token} is not set.")


def transcription_pipeline(
    file_path: Path,
    src_lang: str,
    model_id: str,
    task: str,
    api_key: str,
    speakers: int,
) -> pd.DataFrame:
    """
    Transcribe and diarize the audio file using WhisperX.

    Args:
        file_path (Path): Path to the audio file.
        src_lang (str): Source language code.
        model_id (str): Model ID for WhisperX.
        task (str): Task to perform (transcribe or translate).
        api_key (str): API key for authentication.
        speakers (int): Number of speakers for diarization.

    Returns:
        pd.DataFrame: DataFrame containing the transcribed text and speaker diarization.
    """
    transcriber = WhisperTranscriber(
        file_path=file_path,
        language=src_lang,
        model_id=model_id,
        task=task,
        auth_token=api_key,
    )
    transcriber.transcription()
    return transcriber.diarization(speakers)


def translation_pipeline(df: pd.DataFrame, trg_lang: str) -> pd.DataFrame:
    """
    Translate the transcribed text using a machine translation model.

    Args:
        df (pd.DataFrame): DataFrame containing the transcribed text.
        trg_lang (str): Target language code for translation.

    Returns:
        pd.DataFrame: DataFrame with the translated text.
    """
    translator = Translator()
    translator.detect_language(" ".join(df["text"].astype(str).tolist()))
    df["text"] = df["text"].apply(lambda text: translator.translate(trg_lang, text))
    return df


def summarization_pipeline(
    text: str,
    prompt_lang: str = "German",
    trg_lang: str = "de",
) -> str | None:
    """
    Summarize the given text using a language model and translate the result.

    Args:
        text (str): The text to summarize.
        language (str): The language code of the text. Defaults to "German".
        trg_lang (str): The target language code for the summary. Defaults to "de".

    Returns:
        str | None: The summarized text or None if an error occurs.
    """
    try:
        if not text:
            raise ValueError("Text cannot be empty.")
        prompt = text_summarization_prompt.format(
            language=prompt_lang,
            text=text,
        )

        return call_ollama(prompt=prompt)

    except Exception as e:
        logging.error(f"Error summarizing text: {e}")
        return None


def wordlevel_pipeline(
    data: pd.DataFrame,
    language: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, plt.Figure]:
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
            - matplotlib.figure.Figure: Word cloud figure.
    """
    word_analysis = WordCounter(
        text=" ".join(data["text"].astype(str).tolist()),
        language=language,
    )

    word_analysis.text_to_doc()
    word_analysis.lemmatize_doc()
    word_counts = word_analysis.count_words(n_words=30)
    named_entities = word_analysis.named_entity_recognition()
    noun_sentiment = word_analysis.get_noun_adjectives()
    wordcloud = word_analysis.create_wordcloud()

    return word_counts, named_entities, noun_sentiment, wordcloud


def topics_pipeline(
    data: pd.DataFrame,
    language: str,
) -> list[tuple[str, str]] | None:
    """
    Perform topic modeling analysis.

    Args:
        data (pd.DataFrame): DataFrame with the data to analyze.
        language (str): Language of the text data.

    Returns:
        list[tuple[str, str] | None: List of topic titles and summaries, or None if no topics are found.
    """
    topic_modeling = TopicModeling(
        rows=data["text"].astype(str).tolist(),
        lang_code=language,
    )
    topic_modeling.load_pipeline()
    topic_modeling.fit_topic_model()
    return topic_modeling.summarize_topics()


def toxicity_pipeline(data: pd.DataFrame) -> pd.DataFrame:
    """
    Perform toxicity analysis on the text data.

    Args:
        data (pd.DataFrame): DataFrame containing the text data to analyze.

    Returns:
        pd.DataFrame: DataFrame with an additional column for toxicity scores.
    """
    classifier = ToxClassifier()
    result = classifier.classify_data(data["text"].astype(str).tolist())
    data["toxicity"] = result
    return data
