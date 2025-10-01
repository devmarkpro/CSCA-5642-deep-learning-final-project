from abc import ABC, abstractmethod

from config import AppConfig
from logger.app_logger import get_logger


class App(ABC):
    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = get_logger(config.app_name)

    @abstractmethod
    def run(self):
        pass
