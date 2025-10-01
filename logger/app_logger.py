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
        logger_name: Name of the logger to use (if None, uses the configured app logger)
    """
    if logger_name is None:
        # Import here to avoid circular imports
        from config.logger_config import get_app_name
        logger_name = get_app_name()

    logger = logging.getLogger(logger_name)
    extra = {"color": color.value if color != Colors.DEFAULT else ""}
    logger.log(level, msg, extra=extra, stacklevel=2)
