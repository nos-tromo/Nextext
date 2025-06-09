import logging.config
from pathlib import Path

import yaml


_DEFAULT_YAML = """
version: 1
disable_existing_loggers: False

formatters:
    simple:
        format: "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

handlers:
    console:
        class: logging.StreamHandler
        formatter: simple
        level: INFO
    file:
        class: logging.handlers.RotatingFileHandler
        filename: ".log/nextext.log"
        maxBytes: 5_000_000   # 5 MB
        backupCount: 3
        formatter: simple
        level: DEBUG

root:
    level: INFO
    handlers: [console, file]
"""

__all__ = ["setup_logging"]


def setup_logging(path: str | Path | None = None) -> None:
    """
    Setup logging configuration.

    Args:
        path (str | Path | None, optional): Path to a YAML file with logging configuration.
        If None, uses the default configuration. Defaults to None.
    """
    try:
        if path:
            cfg = yaml.safe_load(Path(path).read_text())
        else:
            cfg = yaml.safe_load(_DEFAULT_YAML)
            logging.config.dictConfig(cfg)
            logging.info("Logging setup successfully with configuration: %s", path or "default")
            logging.debug("Logging configuration: %s", cfg)
            logging.getLogger("nextext").info("Nextext logging initialized.")
    except yaml.YAMLError as e:
        logging.basicConfig(level=logging.INFO)
        logging.error(f"Error loading logging configuration from {path}: {e}")
    except Exception as e:
        logging.basicConfig(level=logging.INFO)
        logging.error(f"Error setting up logging: {e}")
