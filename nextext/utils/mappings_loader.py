import json
import logging
from functools import lru_cache
from pathlib import Path


@lru_cache
def load_mappings(
    file: str,
    mappings_dir: str = "mappings",
) -> dict[str, str]:
    """
    Load mappings from a JSON file.

    Args:
        file (str): Description of the file to load.
        mappings_dir (str): Directory where the mappings files are located. Defaults to "mappings".

    Returns:
        dict[str, str]: Key to value mapping of the JSON input.
    """
    logger = logging.getLogger(__name__)

    try:
        _JSON_PATH = Path(__file__).parent / mappings_dir / file
        logger.info(f"Attempting to load mappings from '{_JSON_PATH}'")
        with open(_JSON_PATH, "r", encoding="utf-8") as f:
            code2name = json.load(f)
        logger.info(f"Successfully loaded mappings with {len(code2name)} entries")
        return code2name
    except FileNotFoundError as e:
        logger.error(f"File '{file}' not found in utils/ directory.")
        raise FileNotFoundError(f"File '{file}' not found in utils/ directory.") from e
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from file '{file}': {e}")
        raise ValueError(f"Error decoding JSON from file '{file}': {e}") from e
