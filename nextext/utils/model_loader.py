"""Utilities for preloading Nextext language and speech models."""

import gc
import importlib
import os
import subprocess
import sys
from collections.abc import Iterable
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path

import nltk
from dotenv import load_dotenv
from gliner import GLiNER
from loguru import logger

from nextext.utils.mappings_loader import load_mappings


load_dotenv()

SPACY_MODEL_DIR = "SPACY_MODEL_DIR"
SPACY_MODEL_DOWNLOAD_BASE_URL = "SPACY_MODEL_DOWNLOAD_BASE_URL"
SPACY_MODEL_PACKAGE_VERSION = "SPACY_MODEL_PACKAGE_VERSION"
DEFAULT_SPACY_MODEL_DIR = Path.home() / ".cache" / "spacy"
DEFAULT_SPACY_MODEL_DOWNLOAD_BASE_URL = (
    "https://github.com/explosion/spacy-models/releases/download"
)
NLTK_RESOURCES = ("punkt_tab", "stopwords")
WHISPER_LANGUAGE_DETECTION_MODEL = "small"
GLINER_MODEL_ID = "urchade/gliner_multi-v2.1"
DIARIZATION_MODEL_ID = "pyannote/speaker-diarization-3.1"
DIARIZATION_DEPENDENCY_IDS = ("pyannote/segmentation-3.0",)


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
        raise RuntimeError(
            f"Unsupported spaCy version '{installed_spacy_version}' for model preload."
        )

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
    return (
        f"{get_spacy_model_download_base_url()}/{model_id}-{model_version}/{wheel_name}"
    )


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


def get_whisper_model_ids(
    whisper_models_file: str = "whisper_models.json",
) -> list[str]:
    """Return the distinct Whisper model IDs used by Nextext.

    Args:
        whisper_models_file (str): Mapping file containing model aliases.

    Returns:
        list[str]: Sorted Whisper model IDs including the detection model.
    """
    whisper_models = load_mappings(whisper_models_file)
    model_ids = set(whisper_models.values())
    model_ids.add(WHISPER_LANGUAGE_DETECTION_MODEL)
    return sorted(model_ids)


def ensure_nltk_resources(resources: Iterable[str] = NLTK_RESOURCES) -> None:
    """Download the NLTK resources required by Nextext.

    Args:
        resources (Iterable[str]): Resource names to fetch.
    """
    for resource in resources:
        nltk.download(resource, quiet=True)
        logger.info("Loaded NLTK resource '{}'.", resource)


def download_spacy_model(model_id: str) -> None:
    """Download a spaCy model into the persistent cache directory.

    Args:
        model_id (str): The name of the spaCy model to download.

    Raises:
        subprocess.CalledProcessError: If the subprocess command fails.
    """
    model_dir = ensure_spacy_model_path()
    if is_spacy_model_cached(model_id, model_dir):
        logger.info("spaCy model '{}' already cached in '{}'.", model_id, model_dir)
        return

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


def _cleanup_torch_resources() -> None:
    """Release transient Torch resources after model preloading."""
    import torch

    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def preload_whisper_model(model_id: str, device: str = "cpu") -> None:
    """Preload an openai-whisper speech model into the local cache.

    Args:
        model_id (str): The Whisper model ID to preload.
        device (str): Device used for the preload step.
    """
    import whisper

    logger.info("Loading Whisper model '{}'.", model_id)
    model = whisper.load_model(model_id, device=device)
    del model
    _cleanup_torch_resources()


def preload_whisper_models(
    model_ids: Iterable[str] | None = None,
    device: str = "cpu",
) -> None:
    """Preload the WhisperX ASR models used by Nextext.

    Args:
        model_ids (Iterable[str] | None): Optional explicit model IDs to
            preload.
        device (str): Device used for the preload step.
    """
    resolved_model_ids = list(model_ids or get_whisper_model_ids())
    for model_id in resolved_model_ids:
        preload_whisper_model(model_id, device=device)


def preload_diarization_model(
    auth_token: str | None,
    device: str = "cpu",
) -> None:
    """Preload the pyannote speaker diarization pipeline.

    Args:
        auth_token (str | None): Hugging Face token used for private gated
            model access.
        device (str): Device used for the preload step.
    """
    if not auth_token:
        logger.warning(
            "Skipping diarization preload. '{}' also depends on {}.",
            DIARIZATION_MODEL_ID,
            ", ".join(DIARIZATION_DEPENDENCY_IDS),
        )
        return

    import torch
    from pyannote.audio import Pipeline as DiarizationPipeline

    from nextext.core.transcription import _configure_torch_safe_globals

    _configure_torch_safe_globals()
    logger.info("Loading diarization model '{}'.", DIARIZATION_MODEL_ID)
    pipeline = DiarizationPipeline.from_pretrained(
        DIARIZATION_MODEL_ID,
        use_auth_token=auth_token,
    )
    pipeline.to(torch.device(device))
    del pipeline
    _cleanup_torch_resources()


def preload_gliner_model(model_id: str = GLINER_MODEL_ID) -> None:
    """Download and cache the GLiNER NER model.

    Args:
        model_id (str): The Hugging Face model ID for GLiNER.
    """
    logger.info("Loading GLiNER model '{}'.", model_id)
    model = GLiNER.from_pretrained(model_id)
    del model
    gc.collect()
    logger.info("GLiNER model '{}' cached.", model_id)


def _get_default_device() -> str:
    """Resolve the preferred device for preload operations.

    Returns:
        str: ``cuda`` when available, otherwise ``cpu``.
    """
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    """Preload Nextext's local language and speech models.

    Raises:
        RuntimeError: If any preload operation fails.
    """
    failures: list[str] = []
    device = _get_default_device()
    logger.info("Preloading Nextext models on device '{}'.", device)

    try:
        ensure_nltk_resources()
    except Exception as exc:
        failures.append(f"nltk resources ({exc})")

    for model_id in get_spacy_model_ids():
        try:
            download_spacy_model(model_id)
        except Exception as exc:
            failures.append(f"spaCy {model_id} ({exc})")

    for model_id in get_whisper_model_ids():
        try:
            preload_whisper_model(model_id, device=device)
        except Exception as exc:
            failures.append(f"Whisper {model_id} ({exc})")

    try:
        preload_diarization_model(os.getenv("HF_HUB_TOKEN"), device=device)
    except Exception as exc:
        failures.append(f"Diarization {DIARIZATION_MODEL_ID} ({exc})")

    try:
        preload_gliner_model()
    except Exception as exc:
        failures.append(f"GLiNER {GLINER_MODEL_ID} ({exc})")

    if failures:
        raise RuntimeError("Failed to preload models: " + "; ".join(failures))

    logger.info("All Nextext preload models are available.")


if __name__ == "__main__":
    main()
