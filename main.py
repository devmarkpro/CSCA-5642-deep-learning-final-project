import os

from dotenv import load_dotenv

import app_logger as l
from config import AppConfig
from config.arg_parser import parse_arguments, create_config_from_args
from eda import EDA
from experiment import Experiment

load_dotenv(dotenv_path="./.env", verbose=True)

APP_NAME = os.environ.get("APP_NAME", "DeepLearningProject")


def main():
    # Parse command line arguments
    args = parse_arguments()

    # Remove log file, if requested and exists
    if args.reset_log_file and os.path.exists(args.log_path):
        os.remove(args.log_path)

    # Setup logging with the parsed arguments
    l.setup(APP_NAME, log_path=args.log_path, log_level=args.log_level)

    # Create AppConfig instance from parsed arguments
    config = create_config_from_args(args)

    l.log(
        f"Starting {args.command} command with config: {config}", color=l.Colors.BLUE)

    # Instantiate and run the appropriate command class
    if args.command == "experiment":
        app = Experiment(config)
    elif args.command == "eda":
        app = EDA(config)
    else:
        raise ValueError(f"Unknown command: {args.command}")

    try:
        app.run()
    except Exception as e:
        l.log(
            f"Error running {args.command} command: {e} , config: {config}", color=l.Colors.RED)
        return 1
    return 0


if __name__ == '__main__':
    main()
