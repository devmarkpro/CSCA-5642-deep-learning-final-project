from config import AppConfig


class WandBLogger:
    def __init__(self, config: AppConfig):
        self.config = config

    def log_metric(self, metric_name: str, metric_value: float, step: int):
        pass
