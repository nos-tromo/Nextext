import logging
import subprocess
import sys

from .lang_maps_loader import load_lang_maps


logger = logging.getLogger(__name__)


def download_model(model_id: str) -> None:
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
    models, _ = load_lang_maps("spacy_models.json")
    for model_id in models.values():
        download_model(model_id)
    logger.info("\n✅ All small models downloaded successfully.")


if __name__ == "__main__":
    main()
