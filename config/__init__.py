"""
Configuration package for the deep learning project.

This package contains:
- AppConfig: Main configuration dataclass
- Command: Enum for command types
- arg_parser: Command line argument parsing utilities
"""

from .config import AppConfig, Command, ArgInfo
from .arg_parser import parse_arguments, create_config_from_args, create_parser, add_subcommand
from . import logger_config

__all__ = [
    "AppConfig",
    "Command", 
    "ArgInfo",
    "parse_arguments",
    "create_config_from_args",
    "create_parser",
    "add_subcommand",
    "logger_config"
]