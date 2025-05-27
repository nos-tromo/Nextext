import logging
import subprocess
import sys

from .lang_utils import load_lang_maps


logger = logging.getLogger(__name__)


def download_model(model_name: str) -> None:
    """
    Download a spaCy model using subprocess.

    Args:
        model_name (str): The name of the spaCy model to download.
    """    
    try:
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", model_name], check=True
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"⚠️ Failed to download {model_name}: {e}")
    except Exception as e:
        logger.error(f"💥 Unexpected error with {model_name}: {e}")


def main() -> None:
    """
    Main function to download all small spaCy models.
    """    
    models, _ = load_lang_maps("spacy_models.json")
    for lang, model in models.items():
        download_model(model)
    logger.info("\n✅ All small models downloaded successfully.")


if __name__ == "__main__":
    main()
