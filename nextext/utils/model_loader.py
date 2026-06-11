"""Utilities for preloading Nextext's spaCy and NLTK language resources.

All model inference runs on external endpoints, so the only assets worth
preloading are the spaCy model packages (word-level analysis) and the NLTK
corpora. Downloads are gated by ``NEXTEXT_OFFLINE`` (offline by default —
see :func:`nextext.utils.env_cfg.is_offline`): airgapped hosts ship the
caches instead of downloading.
"""

# Re-exported for tests that monkeypatch through this module.
import importlib as importlib
import os
import subprocess as subprocess
import sys
from collections.abc import Iterable
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path

import nltk as nltk
from dotenv import load_dotenv
from loguru import logger

from nextext.utils.env_cfg import is_offline
from nextext.utils.mappings_loader import load_mappings

load_dotenv()

SPACY_MODEL_DIR = "SPACY_MODEL_DIR"
SPACY_MODEL_DOWNLOAD_BASE_URL = "SPACY_MODEL_DOWNLOAD_BASE_URL"
SPACY_MODEL_PACKAGE_VERSION = "SPACY_MODEL_PACKAGE_VERSION"
DEFAULT_SPACY_MODEL_DIR = Path.home() / ".cache" / "spacy"
DEFAULT_SPACY_MODEL_DOWNLOAD_BASE_URL = "https://github.com/explosion/spacy-models/releases/download"
NLTK_RESOURCES = ("punkt_tab", "stopwords")


def get_spacy_model_dir() -> Path:
    """Resolve the persistent spaCy model cache directory.

    Returns:
        Path: The path to the spaCy model cache directory.
    """
    configured_dir = os.getenv(SPACY_MODEL_DIR)
    if configured_dir:
        return Path(configured_dir).expanduser()
    return DEFAULT_SPACY_MODEL_DIR


def get_spacy_model_package_version() -> str:
    """Resolve the compatible spaCy model package version.

    Returns:
        str: The compatible model package version.
    """
    configured_version = os.getenv(SPACY_MODEL_PACKAGE_VERSION)
    if configured_version:
        return configured_version

    try:
        installed_spacy_version = package_version("spacy")
    except PackageNotFoundError as exc:
        raise RuntimeError("spaCy must be installed before loading models.") from exc

    version_parts = installed_spacy_version.split(".")
    if len(version_parts) < 2:
        raise RuntimeError(f"Unsupported spaCy version '{installed_spacy_version}' for model preload.")

    return f"{version_parts[0]}.{version_parts[1]}.0"


def get_spacy_model_download_base_url() -> str:
    """Resolve the base URL used for direct spaCy model downloads.

    Returns:
        str: Base URL for spaCy model wheel downloads.
    """
    return os.getenv(
        SPACY_MODEL_DOWNLOAD_BASE_URL,
        DEFAULT_SPACY_MODEL_DOWNLOAD_BASE_URL,
    ).rstrip("/")


def get_spacy_model_download_url(model_id: str) -> str:
    """Build the direct wheel URL for a spaCy model package.

    Args:
        model_id (str): The name of the spaCy model package.

    Returns:
        str: The model wheel URL.
    """
    model_version = get_spacy_model_package_version()
    wheel_name = f"{model_id}-{model_version}-py3-none-any.whl"
    return f"{get_spacy_model_download_base_url()}/{model_id}-{model_version}/{wheel_name}"


def ensure_spacy_model_path(model_dir: Path | None = None) -> Path:
    """Create the persistent model directory and add it to ``sys.path``.

    Args:
        model_dir (Path | None): Optional custom directory for spaCy
            models.

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
        model_dir (Path | None): Optional custom directory for spaCy
            models.

    Returns:
        bool: True if the model is cached, False otherwise.
    """
    resolved_dir = ensure_spacy_model_path(model_dir)
    return (resolved_dir / model_id).exists()


def get_spacy_model_ids(
    spacy_models_file: str = "spacy_models.json",
) -> list[str]:
    """Return the distinct spaCy model package names used by Nextext.

    Args:
        spacy_models_file (str): Mapping file containing language to spaCy
            model definitions.

    Returns:
        list[str]: Sorted spaCy package names.
    """
    spacy_models = load_mappings(spacy_models_file)
    return sorted(set(spacy_models.values()))


def ensure_nltk_resources(resources: Iterable[str] = NLTK_RESOURCES) -> None:
    """Download the NLTK resources required by Nextext.

    In offline mode (``NEXTEXT_OFFLINE``, active by default) downloads are
    skipped; resources already on disk keep working.

    Args:
        resources (Iterable[str]): Resource names to fetch.
    """
    resolved_resources = list(resources)
    if is_offline():
        logger.info(
            "Offline mode: skipping NLTK resource downloads ({}).",
            ", ".join(resolved_resources),
        )
        return
    for resource in resolved_resources:
        nltk.download(resource, quiet=True)
        logger.info("Loaded NLTK resource '{}'.", resource)


def download_spacy_model(model_id: str) -> None:
    """Download a spaCy model into the persistent cache directory.

    Args:
        model_id (str): The name of the spaCy model to download.

    Raises:
        FileNotFoundError: If offline mode is enabled and the model is not
            already cached.
        subprocess.CalledProcessError: If the subprocess command fails.
    """
    model_dir = ensure_spacy_model_path()
    if is_spacy_model_cached(model_id, model_dir):
        logger.info("spaCy model '{}' already cached in '{}'.", model_id, model_dir)
        return

    if is_offline():
        raise FileNotFoundError(
            f"spaCy model '{model_id}' is not cached in '{model_dir}' and offline mode is active. "
            "Run 'load-models' with NEXTEXT_OFFLINE=0 on a connected host, or ship the cache volume."
        )

    download_url = get_spacy_model_download_url(model_id)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(model_dir),
            download_url,
        ],
        check=True,
    )
    importlib.invalidate_caches()
    logger.info("Loaded spaCy model '{}'.", model_id)


def preload_spacy_models(model_ids: Iterable[str] | None = None) -> None:
    """Preload all configured spaCy models.

    Args:
        model_ids (Iterable[str] | None): Optional explicit model IDs to
            preload.
    """
    resolved_model_ids = list(model_ids or get_spacy_model_ids())
    for model_id in resolved_model_ids:
        download_spacy_model(model_id)


def main() -> None:
    """Preload Nextext's spaCy and NLTK language resources.

    Raises:
        RuntimeError: If any preload operation fails.
    """
    failures: list[str] = []
    logger.info("Preloading Nextext language resources (offline={}).", is_offline())

    try:
        ensure_nltk_resources()
    except Exception as exc:
        failures.append(f"nltk resources ({exc})")

    for model_id in get_spacy_model_ids():
        try:
            download_spacy_model(model_id)
        except Exception as exc:
            failures.append(f"spaCy {model_id} ({exc})")

    if failures:
        raise RuntimeError("Failed to preload models: " + "; ".join(failures))

    logger.info("All Nextext preload resources are available.")


if __name__ == "__main__":
    main()
