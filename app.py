from abc import ABC, abstractmethod

from config import AppConfig
from config.app_device import get_device_info
from flicker_dataset import FlickerDataset
from logger import WandBLogger
from logger.app_logger import get_logger


class App(ABC):
    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = get_logger(config.app_name)
        self.device = get_device_info()

        self.logger.info(f"Device: {self.device}")

        self.logger.info(f"initializing datasets")
        self._set_datasets()

        self.logger.info(f"initializing wandb")
        self._set_wandb_logger()

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def get_wandb_config(self) -> tuple[str, list[str]]:
        """
        Return WandB project configuration.
        
        Returns:
            tuple: (project_name, tags_list)
        """
        pass

    def _set_datasets(self):
        self.train_dataset, self.val_dataset, self.test_dataset = FlickerDataset.create_splits(
            config=self.config
        )
        # For EDA, we can use the train dataset as the primary dataset for analysis
        self.full_dataset = self.train_dataset

    def _set_wandb_logger(self):
        project_name, tags = self.get_wandb_config()
        self.wandb_logger = WandBLogger(
            config=self.config,
            project_name=project_name,
            tags=tags)
