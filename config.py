import logging
from dataclasses import dataclass, field, fields
from typing import Any, TypedDict, Dict, Optional


class ArgInfo(TypedDict):
    value: Any
    help: str


@dataclass
class AppConfig:
    seed: int = field(default=42, metadata={"help": "text for help", "command": "all"})
    log_level: int = field(default=logging.INFO, metadata={"help": "text for help", "command": "all"})
    artifacts_folder: str = field(default="./artifacts", metadata={"help": "text for help", "command": "all"})
    use_wandb: bool = field(default=False, metadata={"help": "text for help", "command": "all"})

    epochs: int = field(default=100, metadata={"help": "number of epochs", "command": "experiment"})

    dpi: int = field(float=10, metadata={"help": "dpi for EDA's output plots", "command": "eda"})

    def get_args(self, command: Optional[str] = None) -> Dict[str, ArgInfo]:
        params: Dict[str, ArgInfo] = {}

        for f in fields(self):
            name = f.name
            value = getattr(self, name)
            help_text = f.metadata.get("help", "")
            cmd = f.metadata.get("command", "all")

            if command:
                if cmd == command:
                    params[f"--{name}"] = {"value": value, "help": help_text}
            else:
                params[name] = {"value": value, "help": help_text}
        return params
