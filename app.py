from abc import ABC, abstractmethod

from config import AppConfig


class App(ABC):
    def __init__(self, config: AppConfig):
        self.config = config

    @abstractmethod
    def run(self):
        pass
