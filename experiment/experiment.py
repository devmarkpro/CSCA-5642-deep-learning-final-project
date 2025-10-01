"""
Experiment implementation.

This module contains the Experiment class that handles all machine learning
experiment and training tasks.
"""

import app_logger as l
from app import App
from config import AppConfig


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
        l.log(f"Experiment initialized with config: seed={config.seed}, epochs={config.epochs}", 
              color=l.Colors.BLUE)
    
    def run(self):
        """
        Execute the experiment workflow.
        
        This method implements the main experiment logic including model setup,
        training, evaluation, and result saving.
        """
        l.log(f"Starting experiment workflow with config: {self.config}", color=l.Colors.MAGENTA)
        