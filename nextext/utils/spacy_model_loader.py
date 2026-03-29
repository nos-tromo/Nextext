import importlib
import os
import subprocess
import sys
from pathlib import Path

import nltk
from loguru import logger

from nextext.utils.mappings_loader import load_mappings

SPACY_MODEL_DIR_ENV = "NEXTEXT_SPACY_MODEL_DIR"


nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)


def get_spacy_model_dir() -> Path:
    """Resolve the persistent spaCy model cache directory.

    Returns:
        Path: The path to the spaCy model cache directory.
    """
    configured_dir = os.getenv(SPACY_MODEL_DIR_ENV)
    if configured_dir:
        return Path(configured_dir).expanduser()
    return Path.home() / ".cache" / "nextext" / "spacy"


def ensure_spacy_model_path(model_dir: Path | None = None) -> Path:
    """Create the persistent model directory and add it to ``sys.path``.

    Args:
        model_dir (Path | None, optional): Optional custom directory for spaCy models.
        If None, uses the default directory resolved by `get_spacy_model_dir()`. Defaults to None.

    Returns:
        Path: The path to the spaCy model cache directory.
    """
    resolved_dir = model_dir or get_spacy_model_dir()
    resolved_dir.mkdir(parents=True, exist_ok=True)
    resolved_dir_str = str(resolved_dir)
    if resolved_dir_str not in sys.path:
        sys.path.insert(0, resolved_dir_str)
    return resolved_dir


def is_spacy_model_cached(model_id: str, model_dir: Path | None = None) -> bool:
    """Check whether a spaCy model package exists in the persistent cache.

    Args:
        model_id (str): The name of the spaCy model to check.
        model_dir (Path | None, optional): Optional custom directory for spaCy models.

    Returns:
        bool: True if the model is cached, False otherwise.
    """
    resolved_dir = ensure_spacy_model_path(model_dir)
    return (resolved_dir / model_id).exists()


def download_spacy_model(model_id: str) -> None:
    """Download a spaCy model using subprocess.

    Args:
        model_id (str): The name of the spaCy model to download.

    Raises:
        subprocess.CalledProcessError: If the subprocess command fails.
    """
    model_dir = ensure_spacy_model_path()
    if is_spacy_model_cached(model_id, model_dir):
        logger.info("spaCy model '{}' already cached in '{}'.", model_id, model_dir)
        return

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "spacy",
                "download",
                model_id,
                "--target",
                str(model_dir),
            ],
            check=True,
        )
        importlib.invalidate_caches()
    except subprocess.CalledProcessError as e:
        logger.error("⚠️ Failed to download {}: {}", model_id, e)
        raise
    except Exception as e:
        logger.error("💥 Unexpected error with {}: {}", model_id, e)
        raise


def main() -> None:
    """Main function to download all small spaCy models.

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
