"""
utils/logging.py — Centralized logging setup for tess_pipeline.
"""

from __future__ import annotations

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger under the tess_pipeline namespace."""
    return logging.getLogger(name)


def configure_logging(debug: bool = False) -> None:
    """
    Configure the root tess_pipeline logger.

    Called once from the CLI or from user code.
    """
    level = logging.DEBUG if debug else logging.INFO
    logger = logging.getLogger("tess_pipeline")
    logger.setLevel(level)

    # 1. Stdout Handler
    stdout_handler = None
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            stdout_handler = handler
            break

    if stdout_handler is None:
        stdout_handler = logging.StreamHandler(sys.stdout)
        fmt = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
        stdout_handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
        logger.addHandler(stdout_handler)

    stdout_handler.setLevel(level)

    # 2. File Handler in logs/ folder
    file_handler = None
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            file_handler = handler
            break

    if file_handler is None:
        from pathlib import Path
        import datetime
        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"pipeline_{timestamp}.log"
        file_handler = logging.FileHandler(str(log_file))
        fmt = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
        file_handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(file_handler)

    file_handler.setLevel(logging.DEBUG)  # Always log DEBUG to file for detailed history
