import subprocess
import sys

import nltk
from loguru import logger

from nextext.utils.mappings_loader import load_mappings


nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)


def download_spacy_model(model_id: str) -> None:
    """
    Download a spaCy model using subprocess.

    Args:
        model_id (str): The name of the spaCy model to download.

    Raises:
        subprocess.CalledProcessError: If the subprocess command fails.
    """
    try:
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", model_id], check=True
        )
    except subprocess.CalledProcessError as e:
        logger.error("⚠️ Failed to download {}: {}", model_id, e)
        raise
    except Exception as e:
        logger.error("💥 Unexpected error with {}: {}", model_id, e)
        raise


def main() -> None:
    """
    Main function to download all small spaCy models.

    Raises:
        RuntimeError: If any of the model downloads fail.
    """
    try:
        # Load mappings for spaCy models
        models = load_mappings("spacy_models.json")
        if not models:
            logger.warning("No spaCy models found in the mappings.")
            return

        failed_models: list[str] = []

        # Download each model
        for model_id in models.values():
            try:
                download_spacy_model(model_id)
            except Exception:
                failed_models.append(model_id)

        if failed_models:
            raise RuntimeError(
                "Failed to download spaCy models: " + ", ".join(failed_models)
            )

        logger.info("\n✅ All small models downloaded successfully.")

    except Exception as e:
        logger.exception("An unexpected error occurred: {}", e)
        raise


if __name__ == "__main__":
    main()
