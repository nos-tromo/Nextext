"""LLM-based segment translation via the inference provider abstraction."""

import pycountry
from langdetect import detect
from loguru import logger

from nextext.core.openai_cfg import InferencePipeline
from nextext.utils.mappings_loader import load_mappings


class Translator:
    """Translate transcript text with TranslateGemma over the configured inference provider."""

    def __init__(
        self,
        translation_language_file: str = "translategemma_languages.json",
        inference_pipeline: InferencePipeline | None = None,
    ) -> None:
        """Initialize the Translator.

        Args:
            translation_language_file (str): Name of the JSON mapping file
                that lists supported TranslateGemma languages. Defaults to
                ``"translategemma_languages.json"``.
            inference_pipeline (InferencePipeline | None): Shared inference
                client. A new :class:`InferencePipeline` is created when
                ``None``. Defaults to ``None``.
        """
        self.languages = load_mappings(translation_language_file)
        self.inference_pipeline = inference_pipeline or InferencePipeline()
        self.prompt_template = self.inference_pipeline.load_prompt("translation")
        self.src_lang: str | None = None
        self._warned_vllm_model: bool = False

    def detect_language(self, text: str) -> dict[str, str]:
        """Detect the language of a text.

        Args:
            text (str): The text to detect the language of.

        Returns:
            dict[str, str]: A dictionary containing the detected language name and ISO code.
        """
        self.src_lang = detect(text)
        lang_obj = pycountry.languages.get(alpha_2=self.src_lang)
        src_lang_name = lang_obj.name if lang_obj is not None else ""
        return {"name": src_lang_name, "code": self.src_lang or ""}

    @staticmethod
    def _base_language_code(lang_code: str) -> str:
        """Collapse a locale/script code to its base language code.

        Args:
            lang_code (str): The language code to normalize, e.g. "en-US" or "de-CH".

        Returns:
            str: The base language code, e.g. "en" or "de".
        """
        return lang_code.split("-", 1)[0]

    def _language_name(self, lang_code: str) -> str:
        """Resolve an ISO language code to a human-readable name.

        Args:
            lang_code (str): The ISO 639-1 language code.

        Returns:
            str: The human-readable language name, or the original code if it cannot be resolved.
        """
        mapped = self.languages.get(lang_code)
        if mapped:
            return mapped
        lang_obj = pycountry.languages.get(alpha_2=lang_code)
        return lang_obj.name if lang_obj is not None else lang_code

    def _translation_prompt(self, src_lang: str, trg_lang: str, text: str) -> str:
        """Build the translation prompt expected by TranslateGemma-style instruction tuning.

        Args:
            src_lang (str): The source language code.
            trg_lang (str): The target language code.
            text (str): The text to be translated.

        Returns:
            str: The formatted translation prompt.
        """
        src_name = self._language_name(src_lang)
        trg_name = self._language_name(trg_lang)
        return self.prompt_template.format(
            SOURCE_LANG=src_name,
            SOURCE_CODE=src_lang,
            TARGET_LANG=trg_name,
            TARGET_CODE=trg_lang,
            TEXT=text,
        )

    @staticmethod
    def _vllm_translation_prompt(src_code: str, trg_code: str, text: str) -> str:
        """Build the delimiter-format prompt expected by the vLLM TranslateGemma model.

        The ``Infomaniak-AI/vllm-translategemma-4b-it`` model card specifies
        that language codes and the source text are encoded directly in a
        single user message using literal ``<<<...>>>`` delimiters, with no
        system role.

        Args:
            src_code (str): The ISO 639-1 source language code.
            trg_code (str): The ISO 639-1 target language code.
            text (str): The text to be translated.

        Returns:
            str: The wire-format user-message content for the vLLM backend.
        """
        return f"<<<source>>>{src_code}<<<target>>>{trg_code}<<<text>>>{text}"

    def translate(self, trg_lang: str, text: str, src_lang: str | None = None) -> str:
        """Translate text with the configured translation model.

        Args:
            trg_lang (str): The target language code.
            text (str): The text to be translated.
            src_lang (str | None, optional): The source language code. Auto-detected when None.

        Returns:
            str: The translated text.

        Raises:
            ValueError: If the input text is empty, the target language is unsupported, or the
                source language cannot be determined.
        """
        if not text:
            raise ValueError("Input text cannot be empty.")
        if trg_lang not in self.languages:
            raise ValueError(f"Target language '{trg_lang}' is not supported by the translation pipeline.")

        resolved_src_lang = src_lang or self.src_lang
        if resolved_src_lang is None:
            resolved_src_lang = self.detect_language(text).get("code")
        if not resolved_src_lang:
            raise ValueError("Source language could not be determined.")
        if self._base_language_code(resolved_src_lang) == self._base_language_code(trg_lang):
            return text

        self.src_lang = resolved_src_lang

        if self.inference_pipeline.provider == "vllm":
            translation_model = self.inference_pipeline.translation_model
            if not self._warned_vllm_model and "translategemma" not in translation_model.lower():
                logger.warning(
                    "INFERENCE_PROVIDER=vllm but TRANSLATION_MODEL='{}' does "
                    "not look like a TranslateGemma variant. Proceeding anyway.",
                    translation_model,
                )
                self._warned_vllm_model = True
            vllm_prompt = self._vllm_translation_prompt(
                src_code=resolved_src_lang,
                trg_code=trg_lang,
                text=text,
            )
            return self.inference_pipeline.call_model(
                prompt=vllm_prompt,
                model=translation_model,
                temperature=0.0,
                include_system_prompt=False,
            )

        prompt = self._translation_prompt(
            src_lang=resolved_src_lang,
            trg_lang=trg_lang,
            text=text,
        )
        return self.inference_pipeline.call_model(
            prompt=prompt,
            model=self.inference_pipeline.translation_model,
            temperature=0.0,
            system_prompt=("You are a precise translation engine. Return only the translation text."),
        )
