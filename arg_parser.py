import logging
from dataclasses import dataclass, field, fields
from enum import Enum, auto
from typing import Any, Dict, Optional, TypedDict


class ArgInfo(TypedDict):
    value: Any
    help: str


class Command(Enum):
    ALL = auto()
    EXPERIMENT = auto()
    EDA = auto()


def _level_to_int(level: int | str) -> int:
    """Accept 20 or 'INFO'/'info' etc., return a logging level int or raise ValueError."""
    if isinstance(level, int):
        if level in (logging.CRITICAL, logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG, logging.NOTSET):
            return level
        raise ValueError(f"Invalid numeric log level: {level}")
    # string
    name = level.upper().strip()
    value = getattr(logging, name, None)
    if isinstance(value, int):
        return value
    raise ValueError(f"Invalid string log level: {level!r}")


@dataclass(slots=True, kw_only=True)
class AppConfig:
    # Core
    seed: int = field(default=42, metadata={"help": "Random seed for all components", "command": Command.ALL})
    log_level: int | str = field(
        default=logging.INFO,
        metadata={"help": "Log level (e.g. 10/DEBUG, 20/INFO, ...)", "command": Command.ALL},
    )
    artifacts_folder: str = field(
        default="./artifacts",
        metadata={"help": "Where to store outputs, checkpoints, plots, etc.", "command": Command.ALL},
    )
    use_wandb: bool = field(
        default=False,
        metadata={"help": "Enable Weights & Biases logging", "command": Command.ALL},
    )

    # Experiment
    epochs: int = field(
        default=100,
        metadata={"help": "Number of training epochs", "command": Command.EXPERIMENT},
    )

    # EDA
    dpi: int = field(  # fixed: was `field(float=10, ...)` which is invalid + wrong type
        default=100,
        metadata={"help": "DPI for EDA output plots", "command": Command.EDA},
    )

    # --- lifecycle hooks & helpers ---

    def __post_init__(self) -> None:
        # Normalize/validate log level
        self.log_level = _level_to_int(self.log_level)

        # Basic validations with clear error messages
        if self.seed < 0:
            raise ValueError("seed must be >= 0")
        if not isinstance(self.artifacts_folder, str) or not self.artifacts_folder.strip():
            raise ValueError("artifacts_folder must be a non-empty string")
        if self.epochs <= 0:
            raise ValueError("epochs must be a positive integer")
        if self.dpi <= 0:
            raise ValueError("dpi must be a positive integer")

    def _iter_fields(self, command: Optional[Command]) -> list:
        """Yield dataclass fields filtered by command (or all if None)."""
        if command is None:
            return list(fields(self))
        return [f for f in fields(self) if f.metadata.get("command", Command.ALL) in (command, Command.ALL)]

    def to_dict(self, command: Optional[Command] = None) -> Dict[str, ArgInfo]:
        """
        Return {name: {value, help}} for the selected command group.
        If command is None, include all fields with plain names.
        """
        out: Dict[str, ArgInfo] = {}
        for f in self._iter_fields(command):
            name = f.name
            value = getattr(self, name)
            help_text = f.metadata.get("help", "")
            out[name] = {"value": value, "help": help_text}
        return out

    def to_args(self, command: Optional[Command] = None) -> Dict[str, ArgInfo]:
        """
        Return CLI-style mapping {"--name": {value, help}} for the selected command group.
        If command is None, include all fields.
        """
        out: Dict[str, ArgInfo] = {}
        for f in self._iter_fields(command):
            name = f"--{f.name.replace('_', '-')}"
            value = getattr(self, f.name)
            help_text = f.metadata.get("help", "")
            out[name] = {"value": value, "help": help_text}
        return out
