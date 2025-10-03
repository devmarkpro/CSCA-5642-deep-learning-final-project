"""
Experiment implementation.

This module contains the Experiment class that handles all machine learning
experiment and training tasks.
"""

from app import App
from config import AppConfig
from logger import WandBLogger


class Experiment(App):
    """
    Experiment class for performing machine learning experiments.

    This class inherits from the base App class and implements the run method
    to perform experiment-specific tasks using the provided configuration.
    """

    def __init__(self, config: AppConfig):
        """
        Initialize the Experiment instance with configuration.

        Args:
            config: AppConfig instance containing all configuration parameters
        """
        super().__init__(config)
        self.logger.debug(f"Experiment initialized with config: seed={config.seed}, epochs={config.epochs}")

    def get_wandb_config(self) -> tuple[str, list[str]]:
        """Return experiment-specific WandB configuration."""
        project_name = f"{self.config.app_name}-experiments"
        tags = ["experiment", "training", "flickr30k", "deep-learning", "computer-vision", "captioning"]
        return project_name, tags

    def run(self):
        """
        Execute the experiment workflow.

        This method implements the main experiment logic including model setup,
        training, evaluation, and result saving.
        """
        self.logger.debug(f"Starting experiment workflow with config: {self.config}")
