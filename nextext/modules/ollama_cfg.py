import logging
import os

import ollama
import requests
import torch
from requests.exceptions import RequestException

from nextext.utils.mappings_loader import load_mappings

logger = logging.getLogger(__name__)

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
    try:
        models = load_mappings(filename)
        if not models:
            logger.error("Model file '%s' not found or empty.", filename)
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
        logger.info("Loaded Ollama model: %s", model)
        return model

    except Exception as e:
        logger.error("Error loading ollama model: %s", e)
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
    model = _load_ollama_model()

    if not model:
        logger.error("Failed to load Ollama model.")
        return ""
    if not _get_ollama_health():
        logger.error(
            "Ollama server does not respond. Please ensure it is running and accessible."
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
