import json
import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)


@lru_cache
def load_mappings(
    file: str,
    subdir: str = "mappings",
) -> dict[str, str]:
    """
    Load mappings from a JSON file.

    Args:
        file (str): Description of the file to load.
        subdir (str): Directory where the mappings files are located. Defaults to "mappings".

    Returns:
        dict[str, str]: Key to value mapping of the JSON input.
    """
    try:
        _JSON_PATH = Path(__file__).parent / subdir / file
        logger.info("Attempting to load mappings from '%s'", _JSON_PATH)
        with open(_JSON_PATH, "r", encoding="utf-8") as f:
            code2name = json.load(f)
        logger.info("Successfully loaded mappings with %d entries", len(code2name))
        return code2name
    except FileNotFoundError as e:
        logger.error("File '%s' not found in utils/ directory.", file)
        raise FileNotFoundError(f"File '{file}' not found in utils/ directory.") from e
    except json.JSONDecodeError as e:
        logger.error("Error decoding JSON from file '%s': %s", file, e)
        raise ValueError(f"Error decoding JSON from file '{file}': {e}") from e


def load_and_sort_mappings(file: str) -> tuple[dict[str, str], list[str]]:
    """
    Load language mappings from a JSON file.

    Args:
        file (str): The filename of the JSON file containing language mappings.

    Returns:
        tuple[dict[str, str], list[str]]: A tuple containing a dictionary of language mappings
        and a sorted list of language names.
    """
    try:
        maps = load_mappings(file)
        names = sorted(maps.values())
        return maps, names
    except Exception as e:
        logger.error("Error loading language mappings: %s", e)
        return {}, []


def kv_to_vk(mappings: dict[str, str]) -> dict[str, str]:
    """
    Convert a dictionary from key-value to value-key mapping.

    Args:
        mappings (dict[str, str]): A dictionary with keys as language codes and values as language names.

    Returns:
        dict[str, str]: A dictionary with keys as language names and values as language codes.
    """
    try:
        return {v: k for k, v in mappings.items()}
    except Exception as e:
        logger.error("Error converting mappings: %s", e)
        return {}
