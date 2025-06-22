import logging
from pathlib import Path


def load_font_file(file: str, utils: Path = Path("utils") / "fonts") -> Path:
    """
    Load a font file from the specified path, converting it to an absolute path if necessary.

    Args:
        file (str): The name of the font file to load.
        utils (Path, optional): The directory where the font file is located. Defaults to "utils/fonts".

    Raises:
        FileNotFoundError: If the specified font file does not exist in the given path.
        Exception: If any other error occurs while loading the font file.

    Returns:
        The path to the font file, converted to an absolute path if it was not already.
    """
    font_path = None

    try:
        root = Path(__file__).resolve().parent.parent
        font_path = root / utils / file
        logging.info(f"Loaded font file '{file}' from path '{font_path}'")
        return font_path
    except FileNotFoundError:
        logging.error(f"Font file '{file}' not found in {font_path}.")
        raise FileNotFoundError(f"Font file '{file}' not found in {font_path}.")
    except Exception as e:
        logging.error(f"An error occurred while loading the font file '{file}': {e}")
        raise e
