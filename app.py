from abc import ABC, abstractmethod

from config import AppConfig
from config.app_device import get_device_info
from flicker_dataset import Flicker30kDataset
from logger import WandBLogger
from logger.app_logger import get_logger


class App(ABC):
    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = get_logger(config.app_name)
        self.device = get_device_info()
        self.dataset = Flicker30kDataset(config=self.config)
        self.wandb_logger = WandBLogger(config=self.config)

    @abstractmethod
    def run(self):
        pass
