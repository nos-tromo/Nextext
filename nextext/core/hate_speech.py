"""Hate speech detection agent using an LLM via InferencePipeline."""

import json
import re
from typing import TypedDict

from loguru import logger

from nextext.core.openai_cfg import InferencePipeline


class HateSpeechDetection(TypedDict):
    """Structured result from hate speech detection."""

    hate_speech: bool
    category: str
    confidence: str
    reason: str


class HateSpeechDetector:
    """Detect hate speech in text segments using an LLM.

    Args:
        inference_pipeline (InferencePipeline): Shared inference client.
        max_chars (int): Maximum characters of input text sent for detection. Defaults to 2048.
    """

    def __init__(
        self, inference_pipeline: InferencePipeline, max_chars: int = 2048
    ) -> None:
        self.inference_pipeline = inference_pipeline
        self.max_chars = max_chars
        self.prompt_template = inference_pipeline.load_prompt("hate_speech")

    def detect(self, text: str) -> HateSpeechDetection:
        """Analyse a text segment for hate speech and return a structured result.

        Args:
            text (str): The text to analyse.

        Returns:
            HateSpeechDetection: Structured detection result with hate_speech flag,
                category, confidence, and reason.
        """
        prompt = self.prompt_template.replace("{text}", text[: self.max_chars])
        raw = self.inference_pipeline.call_model(
            prompt=prompt,
            system_prompt="You are a content moderation assistant. Respond only with valid JSON.",
        )
        return _parse_hate_speech_payload(raw)


def _parse_hate_speech_payload(raw: str) -> HateSpeechDetection:
    """Parse an LLM hate speech JSON response with fallbacks for malformed output.

    Args:
        raw (str): Raw string response from the LLM.

    Returns:
        HateSpeechDetection: Parsed and normalised detection result.
    """
    data: dict = {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                logger.warning("Could not parse hate speech response: {}", raw[:200])
                return _default_detection()
        else:
            logger.warning("No JSON found in hate speech response: {}", raw[:200])
            return _default_detection()

    return HateSpeechDetection(
        hate_speech=bool(data.get("hate_speech", False)),
        category=str(data.get("category", "none")).lower(),
        confidence=_normalize_confidence(str(data.get("confidence", "low"))),
        reason=str(data.get("reason", "")),
    )


def _default_detection() -> HateSpeechDetection:
    """Return a safe default detection when parsing fails."""
    return HateSpeechDetection(
        hate_speech=False,
        category="none",
        confidence="low",
        reason="Parse error",
    )


def _normalize_confidence(value: str) -> str:
    """Normalise a confidence string to high/medium/low.

    Args:
        value (str): Raw confidence string from the LLM.

    Returns:
        str: One of "high", "medium", or "low".
    """
    normalized = value.lower().strip()
    if normalized in ("high", "medium", "low"):
        return normalized
    return "low"
