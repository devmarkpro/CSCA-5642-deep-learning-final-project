from app import App
from config import AppConfig
from logger import Colors, WandBLogger

class DataCleansing(App):
    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = Colors.get_colored_logger(__name__)
        self.wandb_logger = WandBLogger(config)

    def run(self):
        self.logger.info("Starting data cleansing...")

        try:
            self._remove_small_images()
        except Exception as e:
            self.logger.error(f"Error during data cleansing: {str(e)}")
            raise
        finally:
            # Close WandB logger
            self.wandb_logger.finish()
    
    def _remove_small_images(self):
        self.logger.info("Removing small images...")
        # find and remove small images from the train dataset
        

