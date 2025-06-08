import json
import logging
from functools import lru_cache
from pathlib import Path


@lru_cache
def load_mappings(
    file: str = "whisper_languages.json",
    mappings_dir: str = "mappings",
) -> tuple[dict[str, str], dict[str, str]]:
    """
    Load mappings from a JSON file.

    Args:
        file (str): Description of the file to load.
        mappings_dir (str): Directory where the mappings files are located.

    Returns:
        tuple[dict[str, str], dict[str, str]]: Key to value and value to key mappings.
    """
    logger = logging.getLogger(__name__)

    try:
        _JSON_PATH = Path(__file__).parent / mappings_dir / file
        logger.info(f"Attempting to load mappings from '{_JSON_PATH}'")
        with open(_JSON_PATH, "r", encoding="utf-8") as f:
            code2name = json.load(f)
        name2code = {v: k for k, v in code2name.items()}
        logger.info(f"Successfully loaded mappings with {len(code2name)} entries")
        return code2name, name2code
    except FileNotFoundError as e:
        logger.error(f"File '{file}' not found in utils/ directory.")
        raise FileNotFoundError(f"File '{file}' not found in utils/ directory.") from e
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from file '{file}': {e}")
        raise ValueError(f"Error decoding JSON from file '{file}': {e}") from e
