"""Inference client configuration for OpenAI-compatible APIs."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

OpenAIClient: Any = None

try:
    from openai import OpenAI as _OpenAIClient
except ImportError:  # pragma: no cover - exercised only when dependency is missing
    pass
else:
    OpenAIClient = _OpenAIClient

PROMPT_DIR: Path = Path(__file__).parent.parent / "utils" / "prompts"


@dataclass
class InferencePipeline:
    """Inference pipeline for OpenAI-compatible chat completions."""

    out_language: str = "German"
    _default_model: str | None = field(default=None, init=False)
    _translation_model: str | None = field(default=None, init=False)
    _client: Any | None = field(default=None, init=False)
    prompt_dir: Path = field(default=PROMPT_DIR, init=False)

    def __post_init__(self) -> None:
        """Load the system prompt at initialisation time."""
        self.sys_prompt = self.load_prompt().format(language=self.out_language)

    @property
    def base_url(self) -> str:
        """Resolve the OpenAI-compatible API base URL from the environment.

        Returns:
            str: The configured base URL, stripped of trailing slashes.
        """
        return os.getenv("OPENAI_API_BASE", "").rstrip("/")

    @property
    def api_key(self) -> str:
        """Resolve the API key for the inference provider.

        Returns:
            str: The configured API key.

        Raises:
            RuntimeError: If ``OPENAI_API_KEY`` is not set.
        """
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY must be set in the environment or .env file."
            )
        return key

    @property
    def client(self) -> Any:
        """Lazily create the OpenAI-compatible client.

        Returns:
            Any: An instance of the OpenAI client configured for the provider.

        Raises:
            RuntimeError: If the 'openai' package is not installed.
        """
        if OpenAIClient is None:
            raise RuntimeError(
                "The 'openai' package is required for inference. Run `uv sync` to install it."
            )
        if self._client is None:
            client_kwargs: dict[str, Any] = {"api_key": self.api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            self._client = OpenAIClient(**client_kwargs)
        return self._client

    def get_health(self) -> bool:
        """Validate that the configured inference provider is reachable.

        Performs a GET request to ``{base_url}/models``. Any HTTP response
        (including auth errors) is treated as reachable; only connection
        failures return False.

        Returns:
            bool: True if the provider is healthy, False otherwise.
        """
        if OpenAIClient is None:
            logger.error("The 'openai' package is not installed.")
            return False
        if not self.base_url:
            logger.error("OPENAI_API_BASE is not configured.")
            return False
        try:
            _ = self.api_key
        except RuntimeError as exc:
            logger.error("Inference configuration error: {}", exc)
            return False
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=5,
            )
            return response.status_code < 500
        except requests.RequestException as exc:
            logger.error("Inference health check failed: {}", exc)
            return False

    @property
    def default_model(self) -> str:
        """Resolve the default chat model for summarization and general tasks.

        Returns:
            str: The default model name.

        Raises:
            RuntimeError: If ``TEXT_MODEL`` is not configured.
        """
        if self._default_model is None:
            configured_model = os.getenv("TEXT_MODEL")
            if not configured_model:
                raise RuntimeError(
                    "TEXT_MODEL must be set in the environment or .env file "
                    "for text analysis."
                )
            self._default_model = configured_model
            logger.info("Loaded model '{}'.", self._default_model)
        return self._default_model

    @property
    def translation_model(self) -> str:
        """Resolve the model used for translation.

        Returns:
            str: The translation model name.

        Raises:
            RuntimeError: If ``TRANSLATION_MODEL`` is not configured.
        """
        if self._translation_model is None:
            configured_model = os.getenv("TRANSLATION_MODEL")
            if not configured_model:
                raise RuntimeError(
                    "TRANSLATION_MODEL must be set in the environment or "
                    ".env file for translation."
                )
            self._translation_model = configured_model
        return self._translation_model

    def load_prompt(self, keyword: str = "system") -> str:
        """Load a prompt from the prompts directory based on the given keyword.

        Args:
            keyword (str): The keyword identifying the prompt file (without extension).

        Returns:
            str: The content of the prompt file.

        Raises:
            FileNotFoundError: If the prompt file for the given keyword does not exist.
        """
        prompt_path = self.prompt_dir / f"{keyword}.txt"
        if not prompt_path.is_file():
            raise FileNotFoundError(f"Prompt file for keyword '{keyword}' not found.")
        with open(prompt_path, "r", encoding="utf-8") as f:
            logger.info("Loaded prompt from '{}'", prompt_path)
            return f.read()

    def call_model(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.1,
        seed: int = 42,
        stop: list[str] | None = None,
        num_predict: int | None = None,
        top_p: float | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """Call the configured inference provider via an OpenAI-compatible chat completions API.

        Args:
            prompt (str): The user prompt to send to the model.
            model (str | None): The model to use. Defaults to ``TEXT_MODEL``.
            temperature (float): Sampling temperature for response generation.
            seed (int): Random seed for reproducibility.
            stop (list[str] | None): Stop tokens to end generation early.
            num_predict (int | None): Maximum number of tokens to generate.
            top_p (float | None): Nucleus sampling parameter.
            system_prompt (str | None): Override the default system prompt for this call.

        Returns:
            str: The generated response from the model.

        Raises:
            RuntimeError: If the configured inference provider is not reachable.
        """
        if not self.get_health():
            raise RuntimeError(
                "Inference provider is not reachable. Please check your configuration."
            )

        request_kwargs: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": [
                {
                    "role": "system",
                    "content": self.sys_prompt
                    if system_prompt is None
                    else system_prompt,
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "seed": seed,
        }
        if stop:
            request_kwargs["stop"] = stop
        if num_predict is not None:
            request_kwargs["max_tokens"] = num_predict
        if top_p is not None:
            request_kwargs["top_p"] = top_p

        response = self.client.chat.completions.create(**request_kwargs)
        content = response.choices[0].message.content
        return content.strip() if isinstance(content, str) else ""
