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
    Initialise logging. If `path` is given load YAML from there,
    otherwise use the baked‑in default.
    """
    if path:
        cfg = yaml.safe_load(Path(path).read_text())
    else:
        cfg = yaml.safe_load(_DEFAULT_YAML)
    Path(".log").mkdir(exist_ok=True)
    logging.config.dictConfig(cfg)
