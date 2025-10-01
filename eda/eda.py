"""
EDA (Exploratory Data Analysis) implementation.

This module contains the EDA class that handles all exploratory data analysis tasks.
"""

from app import App
from config import AppConfig
from logger import app_logger as l


class EDA(App):
    """
    EDA class for performing exploratory data analysis.

    This class inherits from the base App class and implements the run method
    to perform EDA-specific tasks using the provided configuration.
    """

    def __init__(self, config: AppConfig):
        """
        Initialize the EDA instance with configuration.

        Args:
            config: AppConfig instance containing all configuration parameters
        """
        super().__init__(config)
        l.log(f"EDA initialized with config: seed={config.seed}, dpi={config.dpi}",
              color=l.Colors.BLUE)

    def run(self):
        """
        Execute the EDA workflow.

        This method implements the main EDA logic including data loading,
        analysis, and visualization generation.
        """
        l.log(
            f"Starting EDA workflow with config: {self.config}",
            color=l.Colors.GREEN)
