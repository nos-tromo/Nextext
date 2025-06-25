import logging
from typing import Any

import pycountry
import torch
from langdetect import detect
from nltk import sent_tokenize
from pyarabic.araby import sentence_tokenize
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from nextext.utils import load_mappings


class Translator:
    """
    A class to handle translation between MADLAD languages using a pre-trained model.

    Attributes:
        madlad_languages (dict): A dictionary mapping MADLAD language codes to their names.
        logger (logging.Logger): Logger for the Translator class.
        tokenizer (AutoTokenizer): Tokenizer for the translation model.
        model (AutoModelForSeq2SeqLM): The translation model.
        device (torch.device): The device on which the model is loaded (CPU, CUDA, or MPS).
        src_lang (str | None): The source language code detected from the input text.

    Methods:
        __init__(madlad_language_file: str = "madlad_languages.json"): Initializes the Translator class.
        _load_model(model_name: str = "google/madlad400-3b-mt", local_only: bool = True): Loads the translation model and tokenizer.
        detect_language(text: str) -> dict[str, str]: Detects the language of a given text.
        _model_inference(lang: str, text: str, verbose: bool = True) -> str: Uses the model for inference on a given text input.
        translate(trg_lang: str, text: str) -> str: Translates text sentence-wise between any supported MADLAD languages.
    """

    def __init__(
        self,
        madlad_language_file: str = "madlad_languages.json",
        madlad_models_file: str = "madlad_models.json",
        fallback_model: str = "google/madlad400-3b-mt",
    ) -> None:
        """
        Initialize the Translator class. Loads the MADLAD language mapping file and initializes the
        translation model. Sets up the logger and loads the model. The model is loaded from local cache
        if available, otherwise it is downloaded from the Hugging Face Hub. The model is set to run on
        GPU if available, otherwise it falls back to CPU.

        Args:
            madlad_language_file (str): Path to the MADLAD language mapping file.
            madlad_models_file (str): Path to the MADLAD models mapping file.
            fallback_model (str): Fallback model name if no suitable model is found.

        Raises:
            RuntimeError: If the model cannot be loaded from local cache or downloaded.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.languages = load_mappings(madlad_language_file)
        models = load_mappings(madlad_models_file)
        if torch.cuda.is_available():
            model_id = models.get("cuda")
        elif torch.backends.mps.is_available():
            model_id = models.get("mps")
        else:
            model_id = models.get("cpu")
        if model_id is None:
            model_id = fallback_model
        self.tokenizer, self.model, self.device = self._load_model(model_id=model_id)
        self.src_lang: str | None = None

    def _load_model(
        self, model_id: str, local_only: bool = True
    ) -> tuple[Any, Any, torch.device]:
        """
        Loads the translation model and tokenizer.

        Tries to load from local cache first. If not found, downloads from Hugging Face Hub.

        Args:
            model_id (str): The name of the pretrained model.
            local_only (bool): Whether to restrict loading to local files only.

        Raises:
            RuntimeError: If the model cannot be loaded from local cache or downloaded.

        Returns:
            tuple: A tuple containing the tokenizer, model, and device.
        """
        try:
            device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
            torch_dtype = (
                torch.float16 if device.type in ["cuda", "mps"] else torch.float32
            )

            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_id, local_files_only=True
                )
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_id, torch_dtype=torch_dtype, local_files_only=local_only
                ).to(device)
                self.logger.info("✅ Loaded model from local cache.")
            except FileNotFoundError:
                self.logger.info(
                    "⬇️ Model not in local cache — downloading from Hugging Face..."
                )
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_id, torch_dtype=torch_dtype
                ).to(device)

            return tokenizer, model, device
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise RuntimeError(
                "Failed to load translation model. Please check the model name or your internet connection."
            ) from e

    def detect_language(self, text: str) -> dict[str, str]:
        """
        Detect the language of a text.

        Args:
            text (str): Text to be translated.

        Returns:
            dict: A dictionary containing the detected language name and code.
        """
        try:
            self.src_lang = detect(text)
            lang_obj = pycountry.languages.get(alpha_2=self.src_lang)
            src_lang_name = lang_obj.name if lang_obj is not None else ""
            return {"name": src_lang_name, "code": self.src_lang or ""}
        except Exception as e:
            self.logger.error(f"Error detecting language: {e}")
            return {"name": "", "code": "", "flag": ""}

    def _model_inference(self, lang: str, text: str, verbose: bool = True) -> str:
        """
        Use the model for inference on a given text input.

        Args:
            lang (str): Target language code.
            text (str): Text to translate.
            verbose (bool): Set warning if the model's context window is exceeded

        Returns:
            str: The translated text.
        """
        try:
            prompt = f"<2{lang}> {text}"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            max_model_len = 256
            input_ids = inputs.get("input_ids")
            input_len = input_ids.shape[1]
            adjusted_max_length = max_model_len

            if input_len >= max_model_len:
                if verbose:
                    self.logger.warning(
                        f"⚠️ Input length ({input_len} tokens) hits or exceeds max context window ({max_model_len}). Output may be truncated or degraded."
                    )
                    adjusted_max_length = input_len
                else:
                    adjusted_max_length = max_model_len

            outputs = self.model.generate(
                **inputs,
                max_length=adjusted_max_length,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
            return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        except Exception as e:
            self.logger.error(f"Error during model inference: {e}")
            return ""

    def translate(self, trg_lang: str, text: str) -> str:
        """
        Translates text sentence-wise between any supported MADLAD languages.

        Args:
            target_lang (str): Target language code.
            text (str): Text to translate.

        Raises:
            ValueError: If the input text is empty or the target language is not supported.
            RuntimeError: If the translation fails due to an error in the pipeline.

        Returns:
            str: Final translated output.
        """
        try:
            if not text:
                raise ValueError("Input text cannot be empty.")
            if trg_lang not in self.languages:
                raise ValueError(
                    f"Target language '{trg_lang}' is not supported by the translation model."
                )
            if self.src_lang == "ar":
                sentences = sentence_tokenize(text)  # Use pyarabic for Arabic
            # --------- Add sentence segmentation models for other languages here --------- #
            else:
                sentences = sent_tokenize(text)  # Use nltk for other languages
            if not sentences:
                raise ValueError("No sentences found in the input text.")
            return " ".join(
                [
                    self._model_inference(trg_lang, sentence)
                    for sentence in sentences
                    if len(sentence) > 0
                ]
            )
        except Exception as e:
            self.logger.error(f"Error during translation pipeline: {e}")
            raise RuntimeError("Translation failed") from e
