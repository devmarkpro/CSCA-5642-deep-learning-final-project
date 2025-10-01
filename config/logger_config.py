"""
Logger configuration module.

This module contains logger setup and configuration utilities.
"""

import logging
import sys

from pythonjsonlogger.json import JsonFormatter

from logger.app_logger import ColorFormatter, RemoveColorFilter, get_logger

__app_name = None


def setup(app_name: str, log_path: str = "./app.log", log_level: int = logging.INFO) -> None:
    """
    Setup the application logger with console and file handlers.
    
    Args:
        app_name: Name of the application for the logger
        log_path: Path to the log file
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    global __app_name
    __app_name = app_name
    logger = logging.getLogger(__app_name)
    logger.setLevel(log_level)
    logger.propagate = False

    logger.handlers.clear()

    # Console handler with color formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColorFormatter(
        "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    ))
    logger.addHandler(console_handler)

    # File handler with JSON formatting
    file_handler = logging.FileHandler(log_path)
    json_formatter = JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(filename)s %(lineno)d %(message)s",
        json_ensure_ascii=False,
    )
    file_handler.setFormatter(json_formatter)
    file_handler.addFilter(RemoveColorFilter())
    logger.addHandler(file_handler)

    # Create a logger instance for this app and log initialization
    app_log = get_logger(__app_name)
    app_log.debug("Logger initialized")


def get_app_name() -> str:
    """Get the current application name."""
    return __app_name


def set_app_name(app_name: str) -> None:
    """Set the application name."""
    global __app_name
    __app_name = app_name
