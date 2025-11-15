import re
from dataclasses import dataclass, field

from loguru import logger

from nextext.modules.ollama_cfg import OllamaPipeline


@dataclass
class HateSpeechDetector:
    """
    A class to detect hate speech in text using an Ollama model.
    """

    ollama_pipeline: OllamaPipeline = field(default_factory=OllamaPipeline)

    def _parse_binary_label(self, resp: str) -> int:
        """
        Parse the response from the Ollama model to extract a binary label (0 or 1).

        Args:
            resp (str): The response from the Ollama model.

        Returns:
            int: The binary label (0 or 1) indicating hate speech, or -1 on error.
        """
        if resp is None:
            logger.error("Ollama returned None for label")
            return -1

        text = str(resp).splitlines()[0].strip().strip("\"'")
        m = re.fullmatch(r"[01]", text)
        if m:
            return int(text)

        tokens = text.replace(":", " ").replace(",", " ").split()
        for tok in tokens:
            if re.fullmatch(r"[01]", tok):
                return int(tok)

        logger.error("Non-binary Ollama response: {}", resp)
        return -1

    def _run_inference(
        self,
        prompt: str,
        stop: list[str] = ["\n"],
        num_predict: int = 1,
        top_k: int = 1,
        top_p: float = 0.0,
    ) -> int:
        """
        Run inference on the given text using the Ollama model.

        Args:
            prompt (str): The prompt to send to the Ollama model.
            stop (list[str], optional): List of stop sequences. Defaults to ["\n"].
            num_predict (int, optional): Number of tokens to predict. Defaults to 1.
            top_k (int, optional): Top-K sampling parameter. Defaults to 1.
            top_p (float, optional): Top-P sampling parameter. Defaults to 0.0

        Returns:
            int: The binary label (0 or 1) indicating hate speech, or -1 on error.
        """
        response = self.ollama_pipeline.call_ollama_server(
            prompt=prompt,
            stop=stop,
            num_predict=num_predict,
            top_k=top_k,
            top_p=top_p,
        )
        return self._parse_binary_label(response)

    def process(self, data: list[str]) -> list[int]:
        """
        Process a list of text strings to detect hate speech.

        Args:
            data (list[str]): List of text strings to analyze.

        Returns:
            list[int]: List of binary labels (0 or 1) indicating hate speech for each input text.
        """
        results = []
        prompt_template = self.ollama_pipeline.load_prompt("hate")

        for text in data:
            print(f"Processing text: {text[:50]}...")
            prompt = prompt_template.format(text=text)
            label = self._run_inference(prompt)
            results.append(label)
        return results
