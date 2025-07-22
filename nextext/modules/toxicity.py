import logging

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.pipelines import pipeline as hf_pipeline
from transformers.pipelines.base import Pipeline

logger = logging.getLogger(__name__)


class ToxClassifier:
    """
    A class to classify text data for toxicity using a pre-trained model.

    Attributes:
        model_id (str): The model ID for the pre-trained toxicity classifier.
        batch_size (int): The size of batches for processing the data.
        classifier (Pipeline): The classification pipeline.

    Methods:
        _load_pipeline(): Loads the pre-trained model and tokenizer into a pipeline.
        classify_data(data: list[str]) -> list[int]: Performs inference on the text data and returns the results.
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
        self.model_id = model_id
        self.batch_size = (
            128
            if torch.cuda.is_available()
            else 32
            if torch.backends.mps.is_available()
            else 1
        )
        self.classifier = self._load_pipeline()

    def _load_pipeline(self) -> Pipeline | None:
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
            logger.error("Error initializing classification pipeline: %s", e)
            return None

    def classify_data(self, data: list[str], results: list[int] = []) -> list[int]:
        """
        Performs inference on the text data using the pre-trained model and returns the classification results.

        Args:
            data (list[str]): A list of text data to classify.
            results (list[int]): A list to store the classification results. Defaults to an empty list.

        Returns:
            list[int]: A list of classification results (1 for toxic, 0 for non-toxic).
        """
        try:
            n_lines = len(data)
            if n_lines == 0:
                logger.warning("No data provided for classification.")
                return []
            if n_lines < self.batch_size:
                logger.warning(
                    "Data size (%d) is smaller than batch size (%d).",
                    n_lines,
                    self.batch_size,
                )
                self.batch_size = n_lines

            for start_idx in range(0, n_lines, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_lines)
                batch = data[start_idx:end_idx]

                if self.classifier is None:
                    logger.error("Classifier pipeline is not initialized.")
                    return []

                batch_results = self.classifier(batch)

                if not batch_results:
                    logger.warning("Classifier returned an empty list!")
                    continue

                for result in batch_results:
                    if not isinstance(result, dict):
                        logger.warning(
                            "Unexpected classifier output type: %s. Expected a dictionary.",
                            type(result),
                        )
                        continue  # Skip invalid results

                    label = result.get("label", "unknown")
                    score = 1 if label == "toxic" else 0
                    results.append(score)

            if len(results) != len(data):
                logger.error(
                    "Mismatch between number of results and data entries."
                )
                return []

            return results

        except Exception as e:
            logger.error(
                "Error during classification inference: %s", e, exc_info=True
            )
            return []
