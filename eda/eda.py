"""
EDA (Exploratory Data Analysis) implementation.

This module contains the EDA class that handles all exploratory data analysis tasks.
"""

from app import App
from config import AppConfig
from dl_dataset import Dataset
from eda.visualizer import Visualizer
from logger import Colors


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
        self.logger.debug(f"EDA initialized with config: {config}")
        self.ds = Dataset(config)
        self.visualizer = Visualizer(self.ds)

    def run(self):
        """
        Execute the EDA workflow.

        This method implements the main EDA logic including data loading,
        analysis, and visualization generation.
        """
        self.visualizer.show_samples(10)
