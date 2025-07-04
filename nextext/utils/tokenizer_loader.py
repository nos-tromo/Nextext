import logging
import subprocess
import sys

import nltk

from .mappings_loader import load_mappings

logger = logging.getLogger(__name__)


nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)


def download_spacy_model(model_id: str) -> None:
    """
    Download a spaCy model using subprocess.

    Args:
        model_name (str): The name of the spaCy model to download.
    """    
    try:
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", model_id], check=True
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"⚠️ Failed to download {model_id}: {e}")
    except Exception as e:
        logger.error(f"💥 Unexpected error with {model_id}: {e}")


def main() -> None:
    """
    Main function to download all small spaCy models.
    """    
    models = load_mappings("spacy_models.json")
    for model_id in models.values():
        download_spacy_model(model_id)
    logger.info("\n✅ All small models downloaded successfully.")


if __name__ == "__main__":
    main()
