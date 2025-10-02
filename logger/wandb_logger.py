from config import AppConfig
from logger.app_logger import get_logger


class WandBLogger:
    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = get_logger(config.app_name)

    def log_metric(self, metric_name: str, metric_value: float, step: int):
        pass
    # log histogram
    def log_histogram(self, histogram_name: str, histogram_value: float, step: int):

