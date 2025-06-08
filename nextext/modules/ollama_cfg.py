import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

import ollama
import requests
import torch

from nextext.utils import load_mappings


def _is_ollama_running(url: Optional[str] = None) -> bool:
    """
    Check if the ollama server is running.

    Args:
        url (str, optional): The URL of the ollama server. Defaults to None.

    Returns:
        bool: True if the server is running, False otherwise.
    """
    if url is None:
        url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    try:
        response = requests.get(url)
        return response.ok
    except requests.exceptions.RequestException:
        return False


def _ensure_ollama_running() -> None:
    """
    Ensure that the ollama server is running. If not, attempt to start it if outside Docker.
    """
    try:
        if not _is_ollama_running():
            if Path("/.dockerenv").exists():
                raise RuntimeError(
                    "Ollama server is not running inside Docker. Please start it on the host."
                )
            subprocess.Popen(["ollama", "serve"])
    except Exception as e:
        logging.error(f"Error starting ollama server: {e}")
        raise RuntimeError("Failed to start ollama server. Please start it manually.")


def _load_ollama_model(
    filename: str = "ollama_models.json", fallback: str = "gemma3:4b-it-qat"
) -> str | None:
    """
    Load the specified ollama model.

    Args:
        filename (str): The name of the JSON file containing model mappings. Defaults to "ollama_models.json".
        fallback (str): The fallback model to use if no suitable model is found. Defaults to "gemma3:4b-it-qat".

    Returns:
        str | None: The name of the model if loaded successfully, None otherwise.
    """
    logger = logging.getLogger(__name__)

    try:
        models, _ = load_mappings(filename)
        if not models:
            logger.error(f"Model file '{filename}' not found or empty.")
            return None
        model = (
            models.get("cuda")
            if torch.cuda.is_available()
            else models.get("mps")
            if torch.backends.mps.is_available()
            else models.get("cpu", fallback)
        )
        logger.info(f"Loaded Ollama model: {model}")
        return model

    except Exception as e:
        logger.error(f"Error loading ollama model: {e}")
        return None


def call_ollama_server(
    prompt: str,
    num_ctx: int = 32768,
    temperature: float = 0.2,
) -> str | None:
    """
    Call the ollama server with the given model and prompt.

    Args:
        prompt (str): The prompt to send to the model.
        num_ctx (int): The number of context tokens to use. Defaults to 8192.
        temperature (float): The temperature for the model's response. Defaults to 0.2.

    Returns:
        str | None: The response from the model, or None if an error occurred.
    """
    logger = logging.getLogger(__name__)

    model = _load_ollama_model()
    
    if not model:
        logger.error("Failed to load Ollama model.")
        return ""
    if not _is_ollama_running():
        logger.warning("Ollama server not running, attempting to start...")
        _ensure_ollama_running()
        time.sleep(5)
        if not _is_ollama_running():
            logger.error("Failed to start Ollama server.")
            return ""

    try:
        ollama_base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        os.environ["OLLAMA_HOST"] = ollama_base_url
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"num_ctx": num_ctx, "temperature": temperature},
        )
        return response["message"]["content"].strip()
    except Exception as e:
        logger.error(f"Error calling ollama server: {e}")
        return ""


# System prompt
system_prompt = """
You are a highly proficient assistant that strictly follows instructions and provides only the requested output.
Do not include interpretations, comments, or acknowledgments unless explicitly asked.
Do not use confirmation phrases such as "Sure, here it comes:", "Got it.", "Here is the translation:", or similar expressions.
Responses should be generated without any markdown formatting unless specified otherwise.
All your outputs must be in {language} language regardless of the input language.
"""

# Text summarization
text_summarization = """
You are an expert summarizer. Create a concise and coherent summary of the following text, capturing all key points and essential information.

Instructions:
1. Content Coverage: Ensure that the summary includes all main ideas and important details from the original text.
2. Brevity: The summary should be concise, ideally between 100 to 200 words unless specified otherwise.
3. Clarity: Use clear and straightforward language. All your outputs must be in {language} language.
4. No Additional Information: Do not include personal opinions, interpretations, or external information.
5. No Extraneous Information: Do not include any Markdown code blocks, additional formatting, or extraneous information.

Text to Summarize:
"{text}"
"""

# Topic summarization
topic_titles = """
You are an expert for topic modeling that is highly proficient in generating topic titles from raw text.
I have a topic that is described by the following keywords: "{keywords}"
The topic contains the following documents: \n"{docs}"
Based on the above information, generate a short label of the topic of at most 5 words.
"""

topic_summaries = """
You are an expert for topic modeling that is highly proficient in summarizing topics from raw text.
I have a topic that is described by the following title: "{title}"
The topic is described by the following keywords: "{keywords}"
The topic contains the following documents: \n"{docs}"
Based on the above information, create a short summary of the topic.
"""

# Combine prompts with system prompt
text_summarization_prompt = system_prompt + "\n\n" + text_summarization
topic_titles_prompt = system_prompt + "\n\n" + topic_titles
topic_summaries_prompt = system_prompt + "\n\n" + topic_summaries
