import logging
from pathlib import Path


def load_font_file(file: str, path: Path = Path("utils") / "fonts") -> Path:
    """
    Load a font file from the specified path, converting it to an absolute path if necessary.

    Args:
        file (str): The name of the font file to load.
        path (Path, optional): The directory where the font file is located. Defaults to "utils/fonts".

    Raises:
        FileNotFoundError: If the specified font file does not exist in the given path.

    Returns:
        The path to the font file, converted to an absolute path if it was not already.
    """
    try:
        root = Path(__file__).resolve().parent.parent
        logging.info(f"Loading font file '{file}' from path '{path}'")
        return path if path.is_absolute() else root / path / file
    except Exception as e:
        logging.error(f"Error loading font file '{file}': {e}")
        raise FileNotFoundError(f"Font file '{file}' not found in {path}.") from e
