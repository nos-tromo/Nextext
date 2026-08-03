"""JSON mappings loader for Whisper/spaCy/language-code config files."""

import json
from functools import lru_cache
from pathlib import Path
from typing import cast

from loguru import logger


@lru_cache
def load_mappings(
    file: str,
    subdir: str = "mappings",
) -> dict[str, str]:
    """Load mappings from a JSON file.

    Args:
        file (str): Description of the file to load.
        subdir (str): Directory where the mappings files are located. Defaults to "mappings".

    Returns:
        dict[str, str]: Key to value mapping of the JSON input.
    """
    _JSON_PATH = Path(__file__).parent / subdir / file
    logger.info("Attempting to load mappings from '{}'", _JSON_PATH)
    with open(_JSON_PATH, encoding="utf-8") as f:
        code2name = json.load(f)
    logger.info("Successfully loaded mappings from file: {}", file)
    return cast(dict[str, str], code2name)


def language_name_from_code(lang_code: str | None, default: str = "") -> str:
    """Resolve an ISO 639-1 language code to its English language name.

    Backed by the bundled ``whisper_languages.json`` mapping (the language
    space Nextext actually handles), so no external ISO database dependency
    is needed. Locale/script suffixes collapse to the base code
    (``"zh-cn"`` → ``"Chinese"``).

    Args:
        lang_code (str | None): The ISO 639-1 language code, optionally with
            a locale suffix (e.g. ``"en"``, ``"de-CH"``).
        default (str): Value returned when the code is empty or unknown.
            Defaults to ``""``.

    Returns:
        str: The English language name, or ``default`` if unresolvable.
    """
    if not lang_code:
        return default
    code = lang_code.lower()
    names = load_mappings("whisper_languages.json")
    return names.get(code) or names.get(code.split("-", 1)[0]) or default


def load_and_sort_mappings(file: str) -> tuple[dict[str, str], list[str]]:
    """Load language mappings from a JSON file.

    Args:
        file (str): The filename of the JSON file containing language mappings.

    Returns:
        tuple[dict[str, str], list[str]]: A tuple containing a dictionary of language mappings
        and a sorted list of language names.
    """
    maps = load_mappings(file)
    names = sorted(maps.values())
    return maps, names


def kv_to_vk(mappings: dict[str, str]) -> dict[str, str]:
    """Convert a dictionary from key-value to value-key mapping.

    Args:
        mappings (dict[str, str]): A dictionary with keys as language codes and values as language names.

    Returns:
        dict[str, str]: A dictionary with keys as language names and values as language codes.
    """
    return {v: k for k, v in mappings.items()}
