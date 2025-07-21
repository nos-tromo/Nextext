import logging
import os

import ollama
import requests
import torch
from requests.exceptions import RequestException

from nextext.utils.mappings_loader import load_mappings

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


def _load_ollama_model(
    filename: str = "ollama_models.json", fallback: str = "gemma3n:e4b"
) -> str | None:
    """
    Load the specified ollama model.

    Args:
        filename (str): The name of the JSON file containing model mappings. Defaults to "ollama_models.json".
        fallback (str): The fallback model to use if no suitable model is found. Defaults to "gemma3:4b-it-qat".

    Returns:
        str | None: The name of the model if loaded successfully, None otherwise.

    Raises:
        RuntimeError: If the model file is not found or empty, or if there is an error loading the model.
    """
    logger = logging.getLogger(__name__)

    try:
        models = load_mappings(filename)
        if not models:
            logger.error(f"Model file '{filename}' not found or empty.")
            raise RuntimeError(
                f"Model file '{filename}' not found or empty. Please check the file."
            )
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
        raise RuntimeError(
            f"Failed to load ollama model from '{filename}'. Please check the file or your configuration."
        )


def _get_ollama_health(url: str = OLLAMA_HOST) -> bool:
    """
    Perform a health check by querying Ollama's /api/tags endpoint.

    Args:
        url (str, optional): The base URL of the Ollama server. Defaults to environment variable or localhost.

    Returns:
        bool: True if the Ollama server responds with model tags, False otherwise.

    Raises:
        RequestException: If there is an error connecting to the Ollama server.
    """
    try:
        response = requests.get(f"{url}/api/tags", timeout=5)
        return response.status_code == 200 and "models" in response.json()
    except RequestException:
        return False


def call_ollama_server(
    prompt: str,
    num_ctx: int = 131072,
    temperature: float = 0.2,
) -> str:
    """
    Call the ollama server with the given model and prompt.

    Args:
        prompt (str): The prompt to send to the model.
        num_ctx (int): The number of context tokens to use. Defaults to 131072.
        temperature (float): The temperature for the model's response. Defaults to 0.2.

    Returns:
        str: The response from the ollama server, or an empty string if an error occurs.

    Raises:
        RuntimeError: If the ollama model cannot be loaded or if the server is not running
    """
    logger = logging.getLogger(__name__)

    model = _load_ollama_model()

    if not model:
        logger.error("Failed to load Ollama model.")
        return ""
    if not _get_ollama_health():
        logger.error(
            "Ollama server is not healthy. Please ensure it is running and accessible."
        )
        return ""

    try:
        # Ensure environment variable is set for ollama library
        os.environ["OLLAMA_HOST"] = OLLAMA_HOST
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
2. Brevity: The summary should be no longer than 15 sentences.
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
