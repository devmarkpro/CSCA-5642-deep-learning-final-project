import enum
import logging


class Colors(enum.Enum):
    RED = "\033[1;31m"
    GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[1;34m"
    MAGENTA = "\033[1;35m"
    CYAN = "\033[1;36m"
    PURPLE = "\033[1;35m"
    DEFAULT = "\033[0m"


class ColorFormatter(logging.Formatter):
    """Colorize the final formatted string if record.color is present."""

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        color = getattr(record, "color", "")
        return f"{color}{base}{Colors.DEFAULT.value}" if color else base


class RemoveColorFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if hasattr(record, "color"):
            delattr(record, "color")
        return True


def log(msg: object, color: Colors = Colors.DEFAULT, level: int = logging.INFO, logger_name: str = None) -> None:
    """
    Log a message with optional color formatting.

    Args:
        msg: Message to log
        color: Color for console output
        level: Logging level
        logger_name: Name of the logger to use (required if no default logger is set)
    """
    if logger_name is None:
        raise ValueError("logger_name must be provided. Use get_logger() for convenience.")

    logger = logging.getLogger(logger_name)
    extra = {"color": color.value if color != Colors.DEFAULT else ""}
    logger.log(level, msg, extra=extra, stacklevel=2)


class AppLogger:
    """
    A logger class that preserves the original call site information.
    """

    def __init__(self, app_name: str):
        self.app_name = app_name
        self.logger = logging.getLogger(app_name)

    def log(self, msg: object, color: Colors = Colors.DEFAULT, level: int = logging.INFO) -> None:
        """Log a message with color formatting and correct call site information."""
        extra = {"color": color.value if color != Colors.DEFAULT else ""}
        self.logger.log(level, msg, extra=extra, stacklevel=2)

    def debug(self, msg: object, color: Colors = Colors.BLUE) -> None:
        """Log a debug message."""
        extra = {"color": color.value if color != Colors.DEFAULT else ""}
        self.logger.log(logging.DEBUG, msg, extra=extra, stacklevel=2)

    def info(self, msg: object, color: Colors = Colors.DEFAULT) -> None:
        """Log an info message."""
        extra = {"color": color.value if color != Colors.DEFAULT else ""}
        self.logger.log(logging.INFO, msg, extra=extra, stacklevel=2)

    def warning(self, msg: object, color: Colors = Colors.YELLOW) -> None:
        """Log a warning message."""
        extra = {"color": color.value if color != Colors.DEFAULT else ""}
        self.logger.log(logging.WARNING, msg, extra=extra, stacklevel=2)

    def error(self, msg: object, color: Colors = Colors.RED) -> None:
        """Log an error message."""
        extra = {"color": color.value if color != Colors.DEFAULT else ""}
        self.logger.log(logging.ERROR, msg, extra=extra, stacklevel=2)


def get_logger(app_name: str) -> AppLogger:
    """
    Get a logger instance for the given app name.
    
    Args:
        app_name: Name of the application
        
    Returns:
        An AppLogger instance for the specified app name
    """
    return AppLogger(app_name)
