import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import matplotlib.pyplot as plt
import matplotlib.figure

from config import AppConfig
from logger.app_logger import get_logger
import wandb


class WandBLogger:
    """
    Comprehensive WandB logger for EDA, training, and visualization logging.

    This class provides a unified API for logging to Weights & Biases with support for:
    - Training metrics and hyperparameters
    - EDA visualizations and statistics
    - Matplotlib figures and custom images
    - WandB native charts and tables
    - Graceful fallbacks when WandB is not available
    """

    def __init__(self, config: AppConfig, project_name: Optional[str] = None,
                 experiment_name: Optional[str] = None, tags: Optional[List[str]] = None):
        """
        Initialize WandB logger with configuration.

        Args:
            config: AppConfig instance containing all configuration parameters
            project_name: WandB project name (defaults to config.app_name)
            experiment_name: WandB run name (auto-generated if None)
            tags: List of tags for the run
        """
        self.config = config
        self.logger = get_logger(config.app_name)
        self.wandb_run = None
        self.is_initialized = False

        # Set up project details
        self.project_name = project_name or config.app_name
        self.experiment_name = experiment_name
        self.tags = tags or []

        # Initialize WandB if enabled and available
        self._initialize_wandb()

    def _initialize_wandb(self) -> bool:
        """
        Initialize WandB run with proper error handling.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Initialize wandb run
            self.wandb_run = wandb.init(
                project=self.project_name,
                name=self.experiment_name,
                tags=self.tags,
                config=self.config.to_dict(),
                resume="allow"
            )

            self.is_initialized = True
            self.logger.info(
                f"WandB initialized successfully. Project: {self.project_name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize WandB: {e}")
            return False

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log metrics to WandB.

        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number for the metrics
        """
        try:
            wandb.log(metrics, step=step)
            self.logger.debug(f"Logged metrics: {list(metrics.keys())}")
        except Exception as e:
            self.logger.error(f"Failed to log metrics: {e}")

    def log_hyperparameters(self, hyperparams: Dict[str, Any]) -> None:
        """
        Log hyperparameters to WandB config.

        Args:
            hyperparams: Dictionary of hyperparameter names and values
        """
        try:
            wandb.config.update(hyperparams)
            self.logger.debug(
                f"Logged hyperparameters: {list(hyperparams.keys())}")
        except Exception as e:
            self.logger.error(f"Failed to log hyperparameters: {e}")

    def log_training_step(self, epoch: int, step: int, loss: float,
                          additional_metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a training step with loss and optional additional metrics.

        Args:
            epoch: Current epoch number
            step: Current step/batch number
            loss: Training loss value
            additional_metrics: Optional dictionary of additional metrics
        """
        metrics = {
            "epoch": epoch,
            "train/loss": loss,
            "train/step": step
        }

        if additional_metrics:
            for key, value in additional_metrics.items():
                # Add train/ prefix if not already present
                if not key.startswith(("train/", "val/", "test/")):
                    key = f"train/{key}"
                metrics[key] = value

        self.log_metrics(metrics, step=step)

    def log_validation_step(self, epoch: int, val_loss: float,
                            additional_metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        Log validation metrics.

        Args:
            epoch: Current epoch number
            val_loss: Validation loss value
            additional_metrics: Optional dictionary of additional validation metrics
        """
        metrics = {
            "epoch": epoch,
            "val/loss": val_loss
        }

        if additional_metrics:
            for key, value in additional_metrics.items():
                # Add val/ prefix if not already present
                if not key.startswith(("train/", "val/", "test/")):
                    key = f"val/{key}"
                metrics[key] = value

        self.log_metrics(metrics)

    def log_matplotlib_figure(self, figure: matplotlib.figure.Figure, name: str,
                              step: Optional[int] = None, close_figure: bool = True) -> None:
        """
        Log a matplotlib figure to WandB.

        Args:
            figure: Matplotlib figure object
            name: Name for the logged figure
            step: Optional step number
            close_figure: Whether to close the figure after logging (default: True)
        """
        try:
            wandb.log({name: wandb.Image(figure)}, step=step)
            self.logger.debug(f"Logged matplotlib figure: {name}")

            if close_figure:
                plt.close(figure)

        except Exception as e:
            self.logger.error(f"Failed to log matplotlib figure '{name}': {e}")
            if close_figure:
                plt.close(figure)

    def log_image_from_path(self, image_path: Union[str, Path], name: str,
                            caption: Optional[str] = None, step: Optional[int] = None) -> None:
        """
        Log an image from file path to WandB.

        Args:
            image_path: Path to the image file
            name: Name for the logged image
            caption: Optional caption for the image
            step: Optional step number
        """
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                self.logger.error(f"Image file not found: {image_path}")
                return

            wandb_image = wandb.Image(str(image_path), caption=caption)
            wandb.log({name: wandb_image}, step=step)
            self.logger.debug(f"Logged image from path: {name}")

        except Exception as e:
            self.logger.error(f"Failed to log image '{name}' from path: {e}")

    def log_images_grid(self, images: List[Any], names: List[str],
                        captions: Optional[List[str]] = None, step: Optional[int] = None) -> None:
        """
        Log multiple images as a grid to WandB.

        Args:
            images: List of images (can be PIL Images, numpy arrays, or file paths)
            names: List of names for each image
            captions: Optional list of captions for each image
            step: Optional step number
        """
        if len(images) != len(names):
            self.logger.error("Number of images must match number of names")
            return

        if captions and len(captions) != len(images):
            self.logger.error("Number of captions must match number of images")
            return

        try:
            wandb_images = {}
            for i, (image, name) in enumerate(zip(images, names)):
                caption = captions[i] if captions else None
                wandb_images[name] = wandb.Image(image, caption=caption)

            wandb.log(wandb_images, step=step)
            self.logger.debug(f"Logged {len(images)} images as grid")

        except Exception as e:
            self.logger.error(f"Failed to log images grid: {e}")

    def save_and_log_figure(self, figure: matplotlib.figure.Figure, name: str,
                            step: Optional[int] = None, save_local: bool = True) -> None:
        """
        Save figure locally and log to WandB.

        Args:
            figure: Matplotlib figure object
            name: Name for the figure (used for filename and WandB name)
            step: Optional step number
            save_local: Whether to save figure locally in artifacts folder
        """
        try:
            # Save locally if requested
            if save_local:
                artifacts_path = os.path.join(
                    self.config.artifacts_folder, self.wandb_run.name)
                os.makedirs(artifacts_path, exist_ok=True)

                # Create filename with step if provided
                if step is not None:
                    filename = f"{name}_step_{step}.png"
                else:
                    filename = f"{name}.png"

                filepath = os.path.join(artifacts_path, filename)

                figure.savefig(filepath, dpi=self.config.dpi,
                               bbox_inches='tight')
                self.logger.debug(f"Saved figure locally: {filepath}")

            # Log to WandB
            self.log_matplotlib_figure(figure, name, step, close_figure=False)

        except Exception as e:
            self.logger.error(f"Failed to save and log figure '{name}': {e}")
        finally:
            plt.close(figure)

    # === WandB Native Charts and Tables ===

    def log_histogram(self, data: List[float], name: str, step: Optional[int] = None,
                      num_bins: int = 64) -> None:
        """
        Log a histogram using WandB's native histogram chart.

        Args:
            data: List of numerical values
            name: Name for the histogram
            step: Optional step number
            num_bins: Number of bins for the histogram
        """
        try:
            wandb.log({name: wandb.Histogram(
                data, num_bins=num_bins)}, step=step)
            self.logger.debug(f"Logged histogram: {name}")
        except Exception as e:
            self.logger.error(f"Failed to log histogram '{name}': {e}")

    def log_table(self, data: List[List[Any]], columns: List[str], name: str,
                  step: Optional[int] = None) -> None:
        """
        Log a table using WandB's native table format.

        Args:
            data: List of rows, where each row is a list of values
            columns: List of column names
            name: Name for the table
            step: Optional step number
        """
        try:
            table = wandb.Table(data=data, columns=columns)
            wandb.log({name: table}, step=step)
            self.logger.debug(f"Logged table: {name}")
        except Exception as e:
            self.logger.error(f"Failed to log table '{name}': {e}")

    def log_scatter_plot(self, x_data: List[float], y_data: List[float], name: str,
                         x_label: str = "x", y_label: str = "y",
                         step: Optional[int] = None) -> None:
        """
        Log a scatter plot using WandB's native plotting.

        Args:
            x_data: X-axis data points
            y_data: Y-axis data points
            name: Name for the scatter plot
            x_label: Label for x-axis
            y_label: Label for y-axis
            step: Optional step number
        """
        if len(x_data) != len(y_data):
            self.logger.error("X and Y data must have the same length")
            return

        try:
            data = [[x, y] for x, y in zip(x_data, y_data)]
            table = wandb.Table(data=data, columns=[x_label, y_label])

            wandb.log({
                name: wandb.plot.scatter(table, x_label, y_label, title=name)
            }, step=step)

            self.logger.debug(f"Logged scatter plot: {name}")
        except Exception as e:
            self.logger.error(f"Failed to log scatter plot '{name}': {e}")

    def log_line_plot(self, x_data: List[float], y_data: List[float], name: str,
                      x_label: str = "x", y_label: str = "y",
                      step: Optional[int] = None) -> None:
        """
        Log a line plot using WandB's native plotting.

        Args:
            x_data: X-axis data points
            y_data: Y-axis data points
            name: Name for the line plot
            x_label: Label for x-axis
            y_label: Label for y-axis
            step: Optional step number
        """
        if len(x_data) != len(y_data):
            self.logger.error("X and Y data must have the same length")
            return

        try:
            data = [[x, y] for x, y in zip(x_data, y_data)]
            table = wandb.Table(data=data, columns=[x_label, y_label])

            wandb.log({
                name: wandb.plot.line(table, x_label, y_label, title=name)
            }, step=step)

            self.logger.debug(f"Logged line plot: {name}")
        except Exception as e:
            self.logger.error(f"Failed to log line plot '{name}': {e}")

    def log_confusion_matrix(self, y_true: List[Any], y_pred: List[Any],
                             class_names: List[str], name: str = "confusion_matrix",
                             step: Optional[int] = None) -> None:
        """
        Log a confusion matrix using WandB's native plotting.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            name: Name for the confusion matrix
            step: Optional step number
        """
        try:
            wandb.log({
                name: wandb.plot.confusion_matrix(
                    y_true=y_true,
                    preds=y_pred,
                    class_names=class_names
                )
            }, step=step)

            self.logger.debug(f"Logged confusion matrix: {name}")
        except Exception as e:
            self.logger.error(f"Failed to log confusion matrix '{name}': {e}")

    # === EDA-Specific Methods ===

    def log_dataset_statistics(self, stats: Dict[str, Any], prefix: str = "dataset") -> None:
        """
        Log dataset statistics for EDA as a table.

        Args:
            stats: Dictionary containing dataset statistics
            prefix: Prefix for the table name
        """
        try:
            # Convert stats to table format
            table_data = []
            for key, value in stats.items():
                # Format the key to be more readable
                formatted_key = key.replace('_', ' ').title()

                # Format the value based on its type
                if isinstance(value, float):
                    if value < 1:
                        formatted_value = f"{value:.3f}"
                    elif value < 100:
                        formatted_value = f"{value:.1f}"
                    else:
                        formatted_value = f"{value:,.1f}"
                elif isinstance(value, int):
                    formatted_value = f"{value:,}"
                else:
                    formatted_value = str(value)

                table_data.append([formatted_key, formatted_value])

            # Log as table
            table_name = f"{prefix}_statistics"
            self.log_table(
                data=table_data,
                columns=['Metric', 'Value'],
                name=table_name
            )

            self.logger.debug(
                f"Logged dataset statistics as table: {table_name}")
        except Exception as e:
            self.logger.error(f"Failed to log dataset statistics: {e}")

    def log_data_distribution(self, data: List[float], name: str,
                              create_histogram: bool = True,
                              create_summary_stats: bool = True) -> None:
        """
        Log data distribution with histogram and summary statistics.

        Args:
            data: Numerical data to analyze
            name: Name for the distribution analysis
            create_histogram: Whether to create a histogram
            create_summary_stats: Whether to log summary statistics
        """
        try:
            import numpy as np

            # Log histogram if requested
            if create_histogram:
                self.log_histogram(data, f"{name}_histogram")

            # Log summary statistics if requested
            if create_summary_stats:
                stats = {
                    f"{name}/mean": float(np.mean(data)),
                    f"{name}/std": float(np.std(data)),
                    f"{name}/min": float(np.min(data)),
                    f"{name}/max": float(np.max(data)),
                    f"{name}/median": float(np.median(data)),
                    f"{name}/count": len(data)
                }

                # Add percentiles
                percentiles = [25, 75, 90, 95, 99]
                for p in percentiles:
                    stats[f"{name}/p{p}"] = float(np.percentile(data, p))

                self.log_metrics(stats)

            self.logger.debug(f"Logged data distribution for: {name}")

        except Exception as e:
            self.logger.error(f"Failed to log data distribution '{name}': {e}")

    def log_sample_images(self, images: List[Any], captions: Optional[List[str]] = None,
                          name: str = "sample_images", max_images: int = 10) -> None:
        """
        Log a sample of images for EDA visualization.

        Args:
            images: List of images to log
            captions: Optional captions for the images
            name: Name for the logged images
            max_images: Maximum number of images to log
        """
        # Limit number of images
        if len(images) > max_images:
            images = images[:max_images]
            if captions:
                captions = captions[:max_images]

        try:
            wandb_images = []
            for i, image in enumerate(images):
                caption = captions[i] if captions else f"Sample {i+1}"
                wandb_images.append(wandb.Image(image, caption=caption))

            wandb.log({name: wandb_images})
            self.logger.debug(f"Logged {len(images)} sample images")

        except Exception as e:
            self.logger.error(f"Failed to log sample images: {e}")

    # === Utility Methods ===

    def add_tags(self, tags: List[str]) -> None:
        """
        Add tags to the current WandB run.

        Args:
            tags: List of tags to add
        """
        try:
            current_tags = list(wandb.run.tags) if wandb.run.tags else []
            new_tags = list(set(current_tags + tags))  # Remove duplicates
            wandb.run.tags = new_tags
            self.logger.debug(f"Added tags: {tags}")
        except Exception as e:
            self.logger.error(f"Failed to add tags: {e}")

    def log_code_artifact(self, file_paths: List[Union[str, Path]],
                          artifact_name: str = "source_code") -> None:
        """
        Log source code files as WandB artifacts.

        Args:
            file_paths: List of file paths to include in the artifact
            artifact_name: Name for the artifact
        """
        try:
            artifact = wandb.Artifact(artifact_name, type="code")

            for file_path in file_paths:
                path = Path(file_path)
                if path.exists():
                    artifact.add_file(str(path))
                else:
                    self.logger.warning(f"Code file not found: {path}")

            wandb.log_artifact(artifact)
            self.logger.debug(f"Logged code artifact: {artifact_name}")

        except Exception as e:
            self.logger.error(f"Failed to log code artifact: {e}")

    def finish(self) -> None:
        """
        Finish the WandB run and clean up resources.
        """
        try:
            wandb.finish()
            self.is_initialized = False
            self.wandb_run = None
            self.logger.info("WandB run finished successfully")
        except Exception as e:
            self.logger.error(f"Failed to finish WandB run: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically finish the run."""
        self.finish()
