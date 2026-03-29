import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
import torch
from loguru import logger

from nextext.utils.mappings_loader import load_mappings

OpenAIClient: Any = None

try:
    from openai import OpenAI as _OpenAIClient
except ImportError:  # pragma: no cover - exercised only when dependency is missing
    pass
else:
    OpenAIClient = _OpenAIClient

OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
PROMPT_DIR: Path = Path(__file__).parent.parent / "utils" / "prompts"


@dataclass
class InferencePipeline:
    """Inference pipeline for local Ollama or remote OpenAI-compatible chat completions."""

    provider: str = field(
        default_factory=lambda: os.getenv("INFERENCE_PROVIDER", "ollama")
    )
    ollama_host: str = field(default=OLLAMA_HOST, init=False)
    model_file: str = "ollama_models.json"
    translation_model_file: str = "translation_models.json"
    _default_model: str | None = field(default=None, init=False)
    _translation_model: str | None = field(default=None, init=False)
    _client: Any | None = field(default=None, init=False)
    fallback_model: str = "qwen3:8b"
    fallback_translation_model: str = "translategemma"
    prompt_dir: Path = field(default=PROMPT_DIR, init=False)
    out_language: str = "German"

    def __post_init__(self) -> None:
        """Post-initialization processing to set up the inference pipeline based on the selected provider."""
        self.provider = self.provider.lower()
        logger.info("Inference provider set to: {}", self.provider)
        self.sys_prompt = self.load_prompt().format(language=self.out_language)

    @property
    def api_key(self) -> str:
        """Resolve the API key required by the OpenAI client.

        Returns:
            str: The API key for the configured inference provider.

        Raises:
            RuntimeError: If the provider is 'openai' and the OPENAI_API_KEY is not set.
        """
        if self.provider == "ollama":
            return (
                os.getenv("OLLAMA_API_KEY") or os.getenv("OPENAI_API_KEY") or "ollama"
            )

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY must be set when INFERENCE_PROVIDER is 'openai'."
            )
        return api_key

    @property
    def base_url(self) -> str | None:
        """Resolve the provider base URL for the OpenAI client.

        Returns:
            str | None: The base URL for the API, or None to use the default.
        """
        custom_base_url = os.getenv("OPENAI_BASE_URL")
        if custom_base_url:
            return custom_base_url
        if self.provider == "ollama":
            return f"{self.ollama_host.rstrip('/')}/v1"
        return None

    @property
    def client(self) -> Any:
        """Lazily create the OpenAI-compatible client.

        Returns:
            Any: An instance of the OpenAI client configured for the selected provider.

        Raises:
            RuntimeError: If the 'openai' package is not installed when required.
        """
        if OpenAIClient is None:
            raise RuntimeError(
                "The 'openai' package is required for inference. Run `uv sync` to install it."
            )
        if self._client is None:
            client_kwargs: dict[str, Any] = {"api_key": self.api_key}
            if self.base_url is not None:
                client_kwargs["base_url"] = self.base_url
            self._client = OpenAIClient(**client_kwargs)
        return self._client

    def get_health(self) -> bool:
        """Validate that the configured inference provider is reachable enough to serve requests.

        Returns:
            bool: True if the provider is healthy, False otherwise.

        Raises:
            RuntimeError: If the 'openai' package is not installed when required.
        """
        if OpenAIClient is None:
            logger.error("The 'openai' package is not installed.")
            return False

        if self.provider == "ollama":
            try:
                response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            except requests.RequestException as exc:
                logger.error("Ollama health check failed: {}", exc)
                return False
            return response.status_code == 200 and "models" in response.json()

        try:
            _ = self.api_key
        except RuntimeError as exc:
            logger.error("OpenAI configuration error: {}", exc)
            return False
        return True

    def _select_model(self, file_name: str, fallback_model: str) -> str:
        """Choose the model based on hardware defaults when running against Ollama.

        Args:
            file_name (str): The JSON file containing model mappings.
            fallback_model (str): The fallback model to use if no mappings are found.

        Returns:
            str: The selected model name.
        """
        models = load_mappings(file_name)
        if not models:
            logger.warning(
                "Model file '{}' not found or empty. Falling back to '{}'.",
                file_name,
                fallback_model,
            )
            return fallback_model

        device_key = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        model_name = models.get(device_key) or models.get("cpu") or fallback_model
        logger.info("Loaded {} model '{}'.", self.provider, model_name)
        return model_name

    @property
    def default_model(self) -> str:
        """Resolve the default chat model for summarization and other general tasks.

        Returns:
            str: The default model name.
        """
        if self._default_model is None:
            if self.provider == "ollama":
                self._default_model = (
                    os.getenv("OLLAMA_MODEL")
                    or os.getenv("INFERENCE_MODEL")
                    or self._select_model(self.model_file, self.fallback_model)
                )
            else:
                self._default_model = (
                    os.getenv("OPENAI_MODEL")
                    or os.getenv("INFERENCE_MODEL")
                    or "gpt-4.1-mini"
                )
        return self._default_model

    @property
    def translation_model(self) -> str:
        """Resolve the model used for translation.

        Returns:
            str: The translation model name.
        """
        if self._translation_model is None:
            if self.provider == "ollama":
                self._translation_model = os.getenv(
                    "TRANSLATION_MODEL"
                ) or self._select_model(
                    self.translation_model_file, self.fallback_translation_model
                )
            else:
                self._translation_model = (
                    os.getenv("OPENAI_TRANSLATION_MODEL")
                    or os.getenv("TRANSLATION_MODEL")
                    or self.default_model
                )
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
        think: bool = False,
        num_ctx: int = 32768,
        temperature: float = 0.1,
        seed: int = 42,
        stop: list[str] | None = None,
        num_predict: int | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """
        Call the configured inference provider via an OpenAI-compatible chat completions API.

        Args:
            prompt (str): The user prompt to send to the model.
            model (str | None): The model to use for this call. If None, the default model for the provider will be used.
            think (bool): Whether to enable "thinking" mode for Ollama, which can improve response quality at the cost of latency. Ignored for other providers.
            num_ctx (int): The context window size to use when "thinking" mode is enabled for Ollama. Ignored for other providers.
            temperature (float): The sampling temperature to use for the response generation.
            seed (int): The random seed to use for reproducibility.
            stop (list[str] | None): A list of stop tokens to end the generation.
            num_predict (int | None): The maximum number of tokens to generate in the response.
            top_k (int | None): The top-k sampling parameter to use for Ollama. Ignored for other providers.
            top_p (float | None): The nucleus sampling parameter to use for the response generation.
            system_prompt (str | None): An optional system prompt to override the default system prompt for this call.

        Returns:
            str: The generated response from the model.

        Raises:
            RuntimeError: If the configured inference provider is not healthy or reachable.
        """
        if not self.get_health():
            raise RuntimeError(
                f"{self.provider.capitalize()} inference is not reachable. Please check your configuration."
            )

        extra_body: dict[str, Any] = {}
        if self.provider == "ollama":
            if think:
                extra_body["think"] = think
            if isinstance(num_ctx, int) and num_ctx > 0:
                extra_body["num_ctx"] = num_ctx
            if top_k is not None:
                extra_body["top_k"] = top_k

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
        if extra_body:
            request_kwargs["extra_body"] = extra_body

        response = self.client.chat.completions.create(**request_kwargs)
        content = response.choices[0].message.content
        return content.strip() if isinstance(content, str) else ""

    def _get_ollama_health(self) -> bool:
        """Backwards-compatible alias retained for existing call sites.

        Returns:
            bool: True if the Ollama server is healthy, False otherwise.
        """
        return self.get_health()

    def call_ollama_server(self, prompt: str, **kwargs: Any) -> str:
        """Backwards-compatible alias retained for existing call sites.

        Returns:
            str: The generated response from the Ollama server.
        """
        return self.call_model(prompt=prompt, **kwargs)


OllamaPipeline = InferencePipeline
