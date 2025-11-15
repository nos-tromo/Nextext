import os
from dataclasses import dataclass, field
from pathlib import Path

import ollama
import requests
import torch
from loguru import logger

from nextext.utils.mappings_loader import load_mappings

OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
PROMPT_DIR: Path = Path(__file__).parent.parent / "utils" / "prompts"


@dataclass
class OllamaPipeline:
    ollama_host: str = field(default=OLLAMA_HOST, init=False)
    model_file: str = "ollama_models.json"
    _ollama_model: str | None = field(default=None, init=False)
    fallback_model: str = "qwen3:8b"
    prompt_dir: Path = field(default=PROMPT_DIR, init=False)
    out_language: str = "German"

    def __post_init__(self):
        """
        Post-initialization to set up the Ollama host and load the system prompt.
        """
        logger.info("Ollama host set to: %s", self.ollama_host)
        self.sys_prompt = self.load_prompt().format(language=self.out_language)

    def _get_ollama_health(self) -> bool:
        """
        Perform a health check by querying Ollama's /api/tags endpoint.

        Returns:
            bool: True if the Ollama server responds with model tags, False otherwise.
        """
        response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
        return response.status_code == 200 and "models" in response.json()

    @property
    def ollama_model(self) -> str:
        """
        Load and return the appropriate Ollama model based on system capabilities.

        Returns:
            str: The name of the Ollama model to use.

        Raises:
            RuntimeError: If the model cannot be determined.
        """
        if self._ollama_model is None:
            models = load_mappings(self.model_file)
            if not models:
                raise RuntimeError(
                    f"Model file '{self.model_file}' not found or empty. Please check the file."
                )
            self._ollama_model = (
                models.get("cuda")
                if torch.cuda.is_available()
                else models.get("mps")
                if torch.backends.mps.is_available()
                else models.get("cpu", self.fallback_model)
            )
            logger.info("Loaded Ollama model: %s", self._ollama_model)

        if self._ollama_model is None:
            self._ollama_model = self.fallback_model
            logger.warning(
                "Could not determine Ollama model from system capabilities. Falling back to '%s'.",
                self.fallback_model,
            )

        return self._ollama_model

    def load_prompt(self, keyword: str = "system") -> str:
        """
        Load a prompt from the prompts directory based on the given keyword.

        Args:
            keyword (str, optional): The keyword to identify the prompt file. Defaults to "system".

        Returns:
            str: The content of the prompt file.

        Raises:
            FileNotFoundError: If the prompt file for the given keyword does not exist.
        """
        prompt_path = self.prompt_dir / f"{keyword}.txt"
        if not prompt_path.is_file():
            raise FileNotFoundError(f"Prompt file for keyword '{keyword}' not found.")
        with open(prompt_path, "r", encoding="utf-8") as f:
            logger.info("Loaded prompt from '%s'", prompt_path)
            return f.read()

    def call_ollama_server(
        self,
        prompt: str,
        think: bool = False,
        num_ctx: int = 32768,
        temperature: float = 0.1,
        seed: int = 42,
        stop: list[str] | None = None,
        num_predict: int | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> str:
        """
        Call the ollama server with the given model and prompt.

        Args:
            prompt (str): The prompt to send to the model.
            think (bool): Whether to enable "think" mode for the model. Defaults to False.
            num_ctx (int): The number of context tokens to use. Defaults to 32768.
            temperature (float): The temperature for the model's response. Defaults to 0.1.
            seed (int): The random seed for the model's response. Defaults to 42.
            stop (list[str]): A list of stop sequences for the model's response. Defaults to None.
            num_predict (int | None): The number of tokens to predict. Defaults to None.
            top_k (int | None): The top_k parameter for the model's response. Defaults to None.
            top_p (float | None): The top_p parameter for the model's response. Defaults to None.

        Returns:
            str: The response from the ollama server, or an empty string if an error occurs.

        Raises:
            RuntimeError: If the Ollama model cannot be loaded or the server is unreachable.
        """
        if not self._get_ollama_health():
            logger.error(
                "Ollama server does not respond. Please ensure it is running and accessible."
            )
            raise RuntimeError(
                "Ollama server is not reachable. Please check your configuration."
            )

        # Ensure environment variable is set for ollama library
        os.environ["OLLAMA_HOST"] = self.ollama_host
        response = ollama.chat(
            model=self.ollama_model,
            think=think,
            messages=[
                {
                    "role": "system",
                    "content": self.sys_prompt,
                },
                {"role": "user", "content": prompt},
            ],
            options={
                **(
                    {"num_ctx": num_ctx}
                    if isinstance(num_ctx, int) and num_ctx > 0
                    else {}
                ),
                "temperature": temperature,
                "seed": seed,
                "stop": stop,
                "num_predict": num_predict,
                "top_k": top_k,
                "top_p": top_p,
            },
        )
        return response["message"]["content"].strip()
