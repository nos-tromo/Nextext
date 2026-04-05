import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()


def set_offline_env() -> None:
    """Log the current offline mode status.

    The actual env vars (HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE, etc.) are set at
    module level immediately after ``load_dotenv()`` so they are available before
    ``huggingface_hub`` / ``transformers`` cache their values at import time.
    This function re-applies them (idempotent) and emits a log message.
    """
    if str(os.getenv("NEXTEXT_OFFLINE", "1")).lower() in {"1", "true", "yes"}:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        logger.info("Set Hugging Face libraries to offline mode.")
    else:
        logger.info("Hugging Face libraries are in online mode.")


set_offline_env()  # Apply offline settings at module load time


@dataclass(frozen=True)
class PathConfig:
    """Dataclass for path configuration."""

    logs: Path
    prompts: Path
    hf_hub_cache: Path


def load_path_env() -> PathConfig:
    """Loads path configuration from environment variables or defaults.

    Returns:
        PathConfig: Dataclass containing path configuration.
        - logs (Path): Path to the logs file.
        - prompts (Path): Path to the prompts directory.
        - hf_hub_cache (Path): Path to the Hugging Face Hub cache directory.
    """
    default_cache_cache: Path = Path.home() / ".cache"
    default_hf_hub_cache: Path = default_cache_cache / "huggingface" / "hub"

    utils_dir: Path = Path(__file__).parent.resolve()
    default_prompts_dir: Path = utils_dir / "prompts"
    project_root: Path = utils_dir.parents[1]
    default_log_dir = project_root / ".logs" / "nextext.log"

    return PathConfig(
        logs=Path(os.getenv("LOG_PATH", default_log_dir)).expanduser(),
        prompts=default_prompts_dir,
        hf_hub_cache=Path(os.getenv("HF_HUB_CACHE", default_hf_hub_cache)).expanduser(),
    )
