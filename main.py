import random

import numpy as np
import torch
from dotenv import load_dotenv

from config.arg_parser import parse_arguments, create_config_from_args
from config.logger_config import setup as setup_logger
from eda import EDA
from experiment import Experiment
from logger.app_logger import get_logger, Colors

load_dotenv(dotenv_path="./.env", verbose=True)


def main():
    # Parse command line arguments
    args = parse_arguments()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create AppConfig instance from parsed arguments
    config = create_config_from_args(args)

    # Setup logging with the config values
    setup_logger(config.app_name, log_path=config.log_path, log_level=config.log_level)

    # Get logger for this app
    logger = get_logger(config.app_name)

    logger.log(
        f"Starting {args.command} command with config: {config}", color=Colors.BLUE
    )

    # Instantiate and run the appropriate command class
    if args.command == "experiment":
        app = Experiment(config)
    elif args.command == "eda":
        app = EDA(config)
    else:
        raise ValueError(f"Unknown command: {args.command}")

    try:
        app.run()
    except ValueError as e:
        logger.error(f"Error running {args.command} command: {e} , config: {config}")
        return 1
    return 0


if __name__ == "__main__":
    main()
