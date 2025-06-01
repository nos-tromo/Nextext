import logging
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure


class FileProcessor:
    """
    FileProcessor is the central class for file processing within Nextext.

    Attributes:
        logger (logging.Logger): Logger for the class.
        filename (str): The name of the input file without extension.
        output_path (Path): The output directory path.

    Methods:
        _setup_directories(file_path: Path, output_dir: Path) -> tuple[str, Path]:
            Sets up necessary directories for file processing.
        write_file_output(data: str | list | tuple | pd.DataFrame | plt.Figure, label: str, target_language: str = "") -> str | list | pd.DataFrame | plt.Figure:
            Writes the provided data to an appropriate output file based on its type (text, list, DataFrame, or plot).
    """

    def __init__(
        self,
        file_path: Path | None = None,
        output_dir: Path = Path("output"),
    ) -> None:
        """
        Initializes the FileProcessor class. Sets up the logger and prepares the output directory for file processing.

        Args:
            file_path (Path | None, optional): The path to the input file.
            output_dir (Path, optional): The directory to save the output files. Defaults to "output".
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        if file_path is None or output_dir is None:
            raise ValueError("file_path and output_dir must not be None")
        self.filename, self.output_path = self._setup_directories(file_path, output_dir)

    def _setup_directories(self, file_path: Path, output_dir: Path) -> tuple[str, Path]:
        """
        Set up the necessary directories for file processing. Create the output subdirectory if it doesn't exist.

        Args:
            file_path (Path): The path to the input file.
            output_dir (Path): The directory to save the output files.

        Returns:
            tuple[str, Path]: A tuple containing the filename (without extension) and the output directory path.
        """
        try:
            filename = file_path.stem  # File name without extension
            output_path = output_dir / filename
            if not output_path.exists():
                output_path.mkdir(parents=True, exist_ok=True)
                self.logger.info(
                    f"Output directory {output_path} does not exist. Creating a new one."
                )
            else:
                self.logger.info(f"Output directory {output_path} already exists.")
            return filename, output_path
        except Exception as e:
            self.logger.error(f"Error processing input file: {e}", exc_info=True)
            raise

    def write_file_output(
        self,
        data: str | list | tuple | pd.DataFrame | Figure,
        label: str,
        target_language: str = "",
    ) -> str | list | tuple | pd.DataFrame | Figure | None:
        """
        Write the provided data to an appropriate output file based on its type (text, list, DataFrame, or plot).

        Args:
            data (str | list | tuple | pd.DataFrame | plt.Figure): The data to be written, which can be a string, list, DataFrame, or matplotlib Figure.
            label (str): The label used to create the file name.
            target_language (str, optional): Optional language code to be appended to the file name. Defaults to "".

        Returns:
            str | list | tuple | pd.DataFrame | plt.Figure | None: The data that was written, which can be a string, list, DataFrame, Figure or None if the data type is unsupported.
        """
        try:
            if isinstance(data, str | list | tuple | pd.DataFrame | Figure):
                # Creating paths for file output
                language_suffix = f"_{target_language}" if target_language else ""
                output_file_path = (
                    self.output_path / f"{self.filename}_{label}{language_suffix}"
                )
                output_file_path_csv = output_file_path.with_suffix(".csv")
                output_file_path_txt = output_file_path.with_suffix(".txt")
                output_file_path_excel = output_file_path.with_suffix(".xlsx")
                output_file_path_png = output_file_path.with_suffix(".png")

                # String file operations
                if isinstance(data, str):
                    with open(f"{output_file_path_txt}", "w", encoding="utf-8") as f:
                        f.write(data)
                # List/tuple file operations
                elif isinstance(data, list | tuple):
                    with open(f"{output_file_path_txt}", "w", encoding="utf-8") as f:
                        for item in data:
                            f.write(f"{item}\n")
                # DataFrame file operations
                elif isinstance(data, pd.DataFrame):
                    data.to_csv(output_file_path_csv, index=False, encoding="utf-8")
                    with pd.ExcelWriter(
                        output_file_path_excel, engine="openpyxl"
                    ) as writer:
                        data.to_excel(writer, index=False, sheet_name="Sheet1")
                elif isinstance(data, Figure):
                    data.savefig(output_file_path_png)

                # File saving notifications
                if output_file_path_csv.exists():
                    self.logger.info(f"Saved output: {output_file_path_csv}")
                if output_file_path_txt.exists():
                    self.logger.info(f"Saved output: {output_file_path_txt}")
                if output_file_path_excel.exists():
                    self.logger.info(f"Saved output: {output_file_path_excel}")
                if output_file_path_png.exists():
                    self.logger.info(f"Saved output: {output_file_path_png}")

            else:
                self.logger.error(
                    f"Data type not supported for writing output: {type(data)}",
                    exc_info=True,
                )

            return data
        except Exception as e:
            self.logger.error(f"Error creating output: {e}", exc_info=True)
            raise
