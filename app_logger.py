import enum
import logging
import sys

from pythonjsonlogger.json import JsonFormatter

_app_name = None


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
    if logger_name is None:
        global _app_name
        logger_name = _app_name
    logger = logging.getLogger(logger_name)
    extra = {"color": color.value if color != Colors.DEFAULT else ""}
    logger.log(level, msg, extra=extra, stacklevel=2)


def setup(app_name, log_path: str = "./app.log", log_level: int = logging.INFO):
    global _app_name
    _app_name = app_name
    logger = logging.getLogger(_app_name)
    logger.setLevel(log_level)
    logger.propagate = False

    logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColorFormatter(
        "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    ))
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path)
    json_formatter = JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(filename)s %(lineno)d %(message)s",
        json_ensure_ascii=False,
    )
    file_handler.setFormatter(json_formatter)
    file_handler.addFilter(RemoveColorFilter())
    logger.addHandler(file_handler)

    log("Logger initialized", color=Colors.GREEN)
