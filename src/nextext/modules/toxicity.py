import logging

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline as hf_pipeline
from transformers.pipelines.base import Pipeline


class ToxClassifier:
    """
    A class to classify text data for toxicity using a pre-trained model.

    Attributes:
        model_id (str): The model ID for the pre-trained toxicity classifier.
        batch_size (int): The size of batches for processing the data.
        classifier (pipeline): The classification pipeline.

    Methods:
        _load_pipeline(): Loads the pre-trained model and tokenizer into a pipeline.
        model_inference(): Performs inference on the text data and returns the results.
    """

    def __init__(
        self,
        model_id: str = "textdetox/xlmr-large-toxicity-classifier",
    ) -> None:
        """
        Initializes the ToxClassifier with the provided data and model ID.

        Args:
            model_id (str, optional): The model ID for the pre-trained toxicity classifier. Defaults to "textdetox/xlmr-large-toxicity-classifier".
        """
        self.logger = logging.getLogger(__name__)
        self.model_id = model_id
        self.batch_size = (
            128
            if torch.cuda.is_available()
            else 32
            if torch.backends.mps.is_available()
            else 1
        )
        self.classifier = self._load_pipeline()

    def _load_pipeline(self) -> Pipeline:
        """
        Loads the pre-trained model and tokenizer into a pipeline for text classification.

        Returns:
            Pipeline | None: The classification pipeline.
        """
        try:
            device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
            model_id = self.model_id
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForSequenceClassification.from_pretrained(model_id).to(
                device
            )
            return hf_pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_length=tokenizer.model_max_length,
                padding=True,
                truncation=True,
            )
        except Exception as e:
            self.logger.error(f"Error initializing classification pipeline: {e}")
            return None

    def classify_data(self, data: list[str]) -> list[int] | None:
        """
        Performs inference on the text data using the pre-trained model and returns the classification results.

        Args:
            data (list[str]): A list of text data to classify.

        Returns:
            list[int] | None: A list of classification results (1 for toxic, 0 for non-toxic) or None if an error occurred.
        """
        try:
            n_lines = len(data)
            results = []

            for start_idx in range(0, n_lines, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_lines)
                batch = data[start_idx:end_idx]

                batch_results = self.classifier(batch)

                if not batch_results:
                    self.logger.warning("Classifier returned an empty list!")
                    continue

                for result in batch_results:
                    if not isinstance(result, dict):
                        self.logger.warning(
                            f"Unexpected classifier output type: {type(result)}. Expected a dictionary."
                        )
                        continue  # Skip invalid results

                    label = result.get("label", "unknown")
                    score = 1 if label == "toxic" else 0
                    results.append(score)

            if len(results) != len(data):
                self.logger.error(
                    "Mismatch between number of results and data entries."
                )
                return None

            return results

        except Exception as e:
            self.logger.error(
                f"Error during classification inference: {e}", exc_info=True
            )
            return None
