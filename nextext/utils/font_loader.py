from pathlib import Path

from loguru import logger


def load_font_file(file: str, utils: Path = Path("utils") / "fonts") -> Path:
    """
    Load a font file from the specified path, converting it to an absolute path if necessary.

    Args:
        file (str): The name of the font file to load.
        utils (Path, optional): The directory where the font file is located. Defaults to "utils/fonts".

    Returns:
        The path to the font file, converted to an absolute path if it was not already.

    Raises:
        FileNotFoundError: If the specified font file does not exist in the given path.
    """
    root = Path(__file__).resolve().parent.parent
    font_path = root / utils / file
    logger.info("Loaded font file '{}' from path '{}'", file, font_path)
    return font_path
