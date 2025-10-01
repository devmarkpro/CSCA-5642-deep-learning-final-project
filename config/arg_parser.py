import argparse
from typing import Optional
from .config import AppConfig, Command


def _add_arguments_for_command(parser: argparse.ArgumentParser, config: AppConfig, target_command: Command) -> None:
    """
    Add arguments for a specific command, including ALL arguments and command-specific arguments.

    Args:
        parser: The ArgumentParser or subparser to add arguments to
        config: AppConfig instance to extract field information from
        target_command: The target command to add arguments for
    """
    from dataclasses import fields

    added_args = set()  # Track added arguments to avoid duplicates

    # Get all fields and filter for this command
    for field in fields(config):
        field_command = field.metadata.get("command", Command.ALL)

        # Include if it's ALL or matches the target command
        if field_command == Command.ALL or field_command == target_command:
            arg_name = f"--{field.name.replace('_', '-')}"

            # Skip if already added
            if arg_name in added_args:
                continue

            added_args.add(arg_name)

            value = getattr(config, field.name)
            help_text = field.metadata.get("help", "")

            # Determine the argument type based on the default value
            if isinstance(value, bool):
                # For boolean flags, create store_true/store_false actions
                if value:
                    parser.add_argument(
                        arg_name,
                        action="store_false",
                        dest=field.name,
                        help=f"{help_text}"
                    )
                else:
                    parser.add_argument(
                        arg_name,
                        action="store_true",
                        dest=field.name,
                        help=f"{help_text}"
                    )
            else:
                # For other types, infer from the default value
                arg_type = type(value)
                parser.add_argument(
                    arg_name,
                    type=arg_type,
                    default=value,
                    dest=field.name,
                    help=f"{help_text}"
                )


def _add_config_arguments(parser: argparse.ArgumentParser, config: AppConfig, command: Optional[Command] = None) -> None:
    """
    Add arguments from AppConfig to the parser based on the command.

    Args:
        parser: The ArgumentParser or subparser to add arguments to
        config: AppConfig instance to extract field information from
        command: Command enum to filter arguments (None for all arguments)
    """
    args_dict = config.to_args(command)

    for arg_name, arg_info in args_dict.items():
        value = arg_info["value"]
        help_text = arg_info["help"]

        # Determine the argument type based on the default value
        if isinstance(value, bool):
            # For boolean flags, create store_true/store_false actions
            if value:
                parser.add_argument(
                    arg_name,
                    action="store_false",
                    dest=arg_name.lstrip("-").replace("-", "_"),
                    help=f"{help_text}"
                )
            else:
                parser.add_argument(
                    arg_name,
                    action="store_true",
                    dest=arg_name.lstrip("-").replace("-", "_"),
                    help=f"{help_text}"
                )
        else:
            # For other types, infer from the default value
            arg_type = type(value)
            parser.add_argument(
                arg_name,
                type=arg_type,
                default=value,
                dest=arg_name.lstrip("-").replace("-", "_"),
                help=f"{help_text}"
            )


def create_parser() -> argparse.ArgumentParser:
    """
    Create the main argument parser with subcommands based on AppConfig.

    Returns:
        Configured ArgumentParser with subcommands for EXPERIMENT and EDA
    """
    # Create default config instance to extract field information
    default_config = AppConfig()

    # Main parser
    parser = argparse.ArgumentParser(
        description="Deep Learning Project CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Create subparsers
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        required=True
    )

    # EXPERIMENT subcommand
    experiment_parser = subparsers.add_parser(
        "experiment",
        help="Run experiments and training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Add arguments for experiment (ALL + EXPERIMENT-specific)
    _add_arguments_for_command(
        experiment_parser, default_config, Command.EXPERIMENT)

    # EDA subcommand
    eda_parser = subparsers.add_parser(
        "eda",
        help="Run exploratory data analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Add arguments for EDA (ALL + EDA-specific)
    _add_arguments_for_command(eda_parser, default_config, Command.EDA)

    return parser


def parse_arguments(args: Optional[list] = None) -> argparse.Namespace:
    """
    Parse command line arguments and return the namespace.

    Args:
        args: Optional list of arguments to parse (for testing). If None, uses sys.argv

    Returns:
        Parsed arguments namespace
    """
    parser = create_parser()
    return parser.parse_args(args)


def create_config_from_args(args: argparse.Namespace) -> AppConfig:
    """
    Create an AppConfig instance from parsed command line arguments.

    Args:
        args: Parsed arguments namespace from parse_arguments()

    Returns:
        AppConfig instance with values from command line arguments
    """
    # Convert namespace to dict, excluding the 'command' field
    args_dict = vars(args).copy()
    # Remove command field as it's not part of AppConfig
    args_dict.pop('command', None)

    # Create AppConfig with the parsed arguments
    return AppConfig(**args_dict)


# Convenience function for adding new subcommands in the future
def add_subcommand(subparsers, command_name: str, command_enum: Command, help_text: str) -> argparse.ArgumentParser:
    """
    Add a new subcommand to the parser. This function makes it easy to extend
    the CLI with new commands in the future.

    Args:
        subparsers: The subparsers object from the main parser
        command_name: Name of the command (e.g., "train", "evaluate")
        command_enum: Command enum value for filtering arguments
        help_text: Help text for the subcommand

    Returns:
        The created subparser
    """
    default_config = AppConfig()

    subparser = subparsers.add_parser(
        command_name,
        help=help_text,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Add arguments for the command (ALL + command-specific)
    _add_arguments_for_command(subparser, default_config, command_enum)

    return subparser
