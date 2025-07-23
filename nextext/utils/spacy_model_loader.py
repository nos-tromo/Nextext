import logging
import subprocess
import sys

import nltk

from nextext.utils.mappings_loader import load_mappings

logger = logging.getLogger(__name__)


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
        logger.error("⚠️ Failed to download %s: %s", model_id, e)
    except Exception as e:
        logger.error("💥 Unexpected error with %s: %s", model_id, e)


def main() -> None:
    """
    Main function to download all small spaCy models.
    """
    try:
        # Load mappings for spaCy models
        models = load_mappings("spacy_models.json")
        if not models:
            logger.warning("No spaCy models found in the mappings.")
            return

        # Download each model
        for model_id in models.values():
            download_spacy_model(model_id)

        logger.info("\n✅ All small models downloaded successfully.")

    except Exception as e:
        logger.exception("An unexpected error occurred: %s", e)
        raise


if __name__ == "__main__":
    main()
