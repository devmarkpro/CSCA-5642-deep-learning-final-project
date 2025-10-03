"""
EDA (Exploratory Data Analysis) implementation.

This module contains the EDA class that handles all exploratory data analysis tasks.
"""

import os
import pandas as pd
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from wordcloud import WordCloud
import re

from app import App
from config import AppConfig
from logger import Colors, WandBLogger


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

        plt.style.use('seaborn-v0_8')

        plt.rcParams['figure.dpi'] = config.dpi
        plt.rcParams['savefig.dpi'] = config.dpi

    def get_wandb_config(self) -> tuple[str, list[str]]:
        """Return EDA-specific WandB configuration."""
        project_name = f"{self.config.app_name}-eda"
        tags = ["eda", "exploratory-data-analysis",
                "flickr30k", "analysis", "visualization"]
        return project_name, tags

    def run(self):
        """
        Execute the comprehensive EDA workflow.

        This method implements the main EDA logic including:
        - Dataset overview and statistics
        - Caption analysis (length, word frequency, etc.)
        - Image analysis (dimensions, aspect ratios, etc.)
        - Data split analysis
        - Sample visualizations
        """
        self.logger.info("Starting comprehensive EDA for Flickr30K dataset")

        try:
            self._analyze_dataset_overview()
            self._analyze_captions()
            self._analyze_images()
            self._analyze_data_splits()
            self._create_sample_visualizations()
            self._generate_summary_report()

            self.logger.info("EDA completed successfully!")

        except Exception as e:
            self.logger.error(f"Error during EDA: {str(e)}")
            raise
        finally:
            # Close WandB logger
            self.wandb_logger.finish()

    def _analyze_dataset_overview(self):
        """Analyze basic dataset statistics and overview."""
        self.logger.info("Analyzing dataset overview...")

        # Get basic statistics
        total_images = len(self.train_dataset) + \
            len(self.val_dataset) + len(self.test_dataset)
        train_size = len(self.train_dataset)
        val_size = len(self.val_dataset)
        test_size = len(self.test_dataset)

        # Load full captions data for analysis
        captions_df = pd.read_csv(
            self.config.dataset_caption_file_path, skipinitialspace=True)
        # Clean data - remove NaN values
        captions_df = captions_df.dropna(subset=['caption', 'image'])
        total_captions = len(captions_df)
        unique_images = captions_df['image'].nunique()
        avg_captions_per_image = total_captions / unique_images

        # Create overview statistics
        overview_stats = {
            'total_images': total_images,
            'unique_images': unique_images,
            'total_captions': total_captions,
            'avg_captions_per_image': avg_captions_per_image,
            'train_images': train_size,
            'val_images': val_size,
            'test_images': test_size,
            'train_ratio': train_size / total_images,
            'val_ratio': val_size / total_images,
            'test_ratio': test_size / total_images
        }

        # Log to WandB
        self.wandb_logger.log_dataset_statistics(overview_stats)

        # Create overview visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Flickr30K Dataset Overview',
                     fontsize=16, fontweight='bold')

        # Split distribution pie chart
        sizes = [train_size, val_size, test_size]
        labels = ['Train', 'Validation', 'Test']
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        ax1.pie(sizes, labels=labels, colors=colors,
                autopct='%1.1f%%', startangle=90)
        ax1.set_title('Data Split Distribution')

        # Images vs Captions bar chart
        categories = ['Unique Images', 'Total Captions']
        values = [unique_images, total_captions]
        bars = ax2.bar(categories, values, color=['#ff7f0e', '#2ca02c'])
        ax2.set_title('Images vs Captions Count')
        ax2.set_ylabel('Count')
        for bar, value in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                     f'{value:,}', ha='center', va='bottom')

        # Captions per image distribution
        captions_per_image = captions_df.groupby('image').size()
        ax3.hist(captions_per_image, bins=range(1, captions_per_image.max()+2),
                 alpha=0.7, color='skyblue', edgecolor='black')
        ax3.set_title('Distribution of Captions per Image')
        ax3.set_xlabel('Number of Captions')
        ax3.set_ylabel('Number of Images')
        ax3.axvline(avg_captions_per_image, color='red', linestyle='--',
                    label=f'Mean: {avg_captions_per_image:.1f}')
        ax3.legend()

        # Dataset size comparison
        split_names = ['Train', 'Val', 'Test']
        split_sizes = [train_size, val_size, test_size]
        bars = ax4.bar(split_names, split_sizes, color=colors)
        ax4.set_title('Dataset Split Sizes')
        ax4.set_ylabel('Number of Images')
        for bar, size in zip(bars, split_sizes):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(split_sizes)*0.01,
                     f'{size:,}', ha='center', va='bottom')

        plt.tight_layout()
        self.wandb_logger.save_and_log_figure(
            fig, 'dataset_overview')
        plt.close()

        self.logger.info(f"Dataset overview: {unique_images} unique images, {total_captions} captions, "
                         f"{avg_captions_per_image:.1f} captions per image on average")

    def _analyze_captions(self):
        """Analyze caption characteristics and statistics."""
        self.logger.info("Analyzing captions...")

        # Load captions data
        captions_df = pd.read_csv(
            self.config.dataset_caption_file_path, skipinitialspace=True)
        # Clean captions - remove NaN values and ensure all are strings
        captions_df = captions_df.dropna(subset=['caption'])
        captions = [str(caption)
                    for caption in captions_df['caption'].tolist()]

        # Calculate caption statistics
        caption_lengths = [len(caption) for caption in captions]
        caption_word_counts = [len(caption.split()) for caption in captions]

        # Word frequency analysis
        all_words = []
        for caption in captions:
            # Clean and tokenize
            words = re.findall(r'\b[a-zA-Z]+\b', caption.lower())
            all_words.extend(words)

        word_freq = Counter(all_words)
        most_common_words = word_freq.most_common(50)

        # Caption statistics
        caption_stats = {
            'total_captions': len(captions),
            'avg_caption_length': np.mean(caption_lengths),
            'median_caption_length': np.median(caption_lengths),
            'min_caption_length': min(caption_lengths),
            'max_caption_length': max(caption_lengths),
            'std_caption_length': np.std(caption_lengths),
            'avg_words_per_caption': np.mean(caption_word_counts),
            'median_words_per_caption': np.median(caption_word_counts),
            'min_words_per_caption': min(caption_word_counts),
            'max_words_per_caption': max(caption_word_counts),
            'unique_words': len(set(all_words)),
            'total_words': len(all_words)
        }

        # Log statistics as table
        self.wandb_logger.log_dataset_statistics(
            caption_stats, prefix="captions")

        # Create caption analysis visualizations
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Caption Analysis', fontsize=16, fontweight='bold')

        # Caption length distribution
        ax1.hist(caption_lengths, bins=50, alpha=0.7,
                 color='lightblue', edgecolor='black')
        ax1.axvline(np.mean(caption_lengths), color='red', linestyle='--',
                    label=f'Mean: {np.mean(caption_lengths):.1f}')
        ax1.axvline(np.median(caption_lengths), color='green', linestyle='--',
                    label=f'Median: {np.median(caption_lengths):.1f}')
        ax1.set_title('Distribution of Caption Lengths (Characters)')
        ax1.set_xlabel('Caption Length (Characters)')
        ax1.set_ylabel('Frequency')
        ax1.legend()

        # Word count distribution
        ax2.hist(caption_word_counts, bins=30, alpha=0.7,
                 color='lightcoral', edgecolor='black')
        ax2.axvline(np.mean(caption_word_counts), color='red', linestyle='--',
                    label=f'Mean: {np.mean(caption_word_counts):.1f}')
        ax2.axvline(np.median(caption_word_counts), color='green', linestyle='--',
                    label=f'Median: {np.median(caption_word_counts):.1f}')
        ax2.set_title('Distribution of Words per Caption')
        ax2.set_xlabel('Number of Words')
        ax2.set_ylabel('Frequency')
        ax2.legend()

        # Top 20 most common words
        top_words = [word for word, count in most_common_words[:20]]
        top_counts = [count for word, count in most_common_words[:20]]
        ax3.barh(range(len(top_words)), top_counts, color='lightgreen')
        ax3.set_yticks(range(len(top_words)))
        ax3.set_yticklabels(top_words)
        ax3.set_title('Top 20 Most Common Words')
        ax3.set_xlabel('Frequency')
        ax3.invert_yaxis()

        # Caption length vs word count scatter
        sample_indices = np.random.choice(
            len(captions), min(5000, len(captions)), replace=False)
        sample_lengths = [caption_lengths[i] for i in sample_indices]
        sample_word_counts = [caption_word_counts[i] for i in sample_indices]

        ax4.scatter(sample_lengths, sample_word_counts,
                    alpha=0.5, color='purple', s=1)
        ax4.set_title('Caption Length vs Word Count')
        ax4.set_xlabel('Caption Length (Characters)')
        ax4.set_ylabel('Word Count')

        # Add correlation coefficient
        correlation = np.corrcoef(sample_lengths, sample_word_counts)[0, 1]
        ax4.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                 transform=ax4.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        self.wandb_logger.save_and_log_figure(
            fig, 'caption_analysis')
        plt.close()

        # Create word cloud
        if len(all_words) > 0:
            wordcloud_text = ' '.join(all_words)
            wordcloud = WordCloud(width=800, height=400, background_color='white',
                                  max_words=100, colormap='viridis').generate(wordcloud_text)

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title('Word Cloud of Most Common Words',
                         fontsize=16, fontweight='bold')

            self.wandb_logger.save_and_log_figure(
                fig, 'word_cloud')
            plt.close()

        self.logger.info(f"Caption analysis: {len(captions)} captions, "
                         f"avg length: {np.mean(caption_lengths):.1f} chars, "
                         f"avg words: {np.mean(caption_word_counts):.1f}")

    def _analyze_images(self):
        """Analyze image characteristics and statistics."""
        self.logger.info("Analyzing images...")

        # Sample images for analysis (to avoid loading all images)
        # sample_size = min(1000, len(self.train_dataset))
        sample_size = len(self.train_dataset)
        sample_indices = np.random.choice(
            len(self.train_dataset), sample_size, replace=False)

        image_stats = {
            'widths': [],
            'heights': [],
            'small_images': [],
            'aspect_ratios': [],
            'file_sizes': []
        }

        for idx in sample_indices:
            try:
                img_file = self.train_dataset.images[idx]
                img_path = os.path.join(
                    self.config.dataset_image_dir, img_file)

                if os.path.exists(img_path):
                    # Get image dimensions
                    with Image.open(img_path) as img:
                        width, height = img.size
                        image_stats['widths'].append(width)
                        image_stats['heights'].append(height)
                        image_stats['aspect_ratios'].append(width / height)

                        if width < self.config.min_image_size or height < self.config.min_image_size:
                            image_stats['small_images'].append(True)
                        else:
                            image_stats['small_images'].append(False)

                    # Get file size
                    file_size = os.path.getsize(img_path) / 1024  # KB
                    image_stats['file_sizes'].append(file_size)

            except Exception as e:
                self.logger.warning(
                    f"Error processing image {img_file}: {str(e)}")
                continue

        if not image_stats['widths']:
            self.logger.warning("No images could be processed for analysis")
            return

        # Calculate statistics
        stats_summary = {
            'sample_size': len(image_stats['widths']),
            'avg_width': np.mean(image_stats['widths']),
            'avg_height': np.mean(image_stats['heights']),
            'avg_aspect_ratio': np.mean(image_stats['aspect_ratios']),
            'avg_file_size_kb': np.mean(image_stats['file_sizes']),
            'min_width': min(image_stats['widths']),
            'max_width': max(image_stats['widths']),
            'min_height': min(image_stats['heights']),
            'max_height': max(image_stats['heights']),
            'min_aspect_ratio': min(image_stats['aspect_ratios']),
            'max_aspect_ratio': max(image_stats['aspect_ratios']),
            'small_images': sum(image_stats['small_images'])
        }

        # Log statistics
        self.wandb_logger.log_dataset_statistics(
            stats_summary, prefix="images")

        # Create image analysis visualizations
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Image Analysis', fontsize=16, fontweight='bold')

        # Width and height distributions
        ax1.hist(image_stats['widths'], bins=30,
                 alpha=0.7, label='Width', color='lightblue')
        ax1.hist(image_stats['heights'], bins=30,
                 alpha=0.7, label='Height', color='lightcoral')
        ax1.set_title('Image Dimensions Distribution')
        ax1.set_xlabel('Pixels')
        ax1.set_ylabel('Frequency')
        ax1.legend()

        # Aspect ratio distribution
        ax2.hist(image_stats['aspect_ratios'], bins=30,
                 alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.axvline(np.mean(image_stats['aspect_ratios']), color='red', linestyle='--',
                    label=f'Mean: {np.mean(image_stats["aspect_ratios"]):.2f}')
        ax2.set_title('Aspect Ratio Distribution')
        ax2.set_xlabel('Aspect Ratio (Width/Height)')
        ax2.set_ylabel('Frequency')
        ax2.legend()

        # Width vs Height scatter plot
        ax3.scatter(
            image_stats['widths'], image_stats['heights'], alpha=0.6, s=10, color='purple')
        ax3.set_title('Image Width vs Height')
        ax3.set_xlabel('Width (pixels)')
        ax3.set_ylabel('Height (pixels)')

        # Add diagonal line for square images
        max_dim = max(max(image_stats['widths']), max(image_stats['heights']))
        ax3.plot([0, max_dim], [0, max_dim], 'r--',
                 alpha=0.5, label='Square (1:1)')
        ax3.legend()

        # File size distribution
        ax4.hist(image_stats['file_sizes'], bins=30,
                 alpha=0.7, color='orange', edgecolor='black')
        ax4.axvline(np.mean(image_stats['file_sizes']), color='red', linestyle='--',
                    label=f'Mean: {np.mean(image_stats["file_sizes"]):.1f} KB')
        ax4.set_title('File Size Distribution')
        ax4.set_xlabel('File Size (KB)')
        ax4.set_ylabel('Frequency')
        ax4.legend()

        plt.tight_layout()
        self.wandb_logger.save_and_log_figure(
            fig, 'image_analysis')
        plt.close()

        self.logger.info(f"Image analysis: {len(image_stats['widths'])} images analyzed, "
                         f"avg dimensions: {np.mean(image_stats['widths']):.0f}x{np.mean(image_stats['heights']):.0f}, "
                         f"avg aspect ratio: {np.mean(image_stats['aspect_ratios']):.2f}")

    def _analyze_data_splits(self):
        """Analyze the characteristics of different data splits."""
        self.logger.info("Analyzing data splits...")

        splits_info = {
            'train': {'dataset': self.train_dataset, 'size': len(self.train_dataset)},
            'val': {'dataset': self.val_dataset, 'size': len(self.val_dataset)},
            'test': {'dataset': self.test_dataset, 'size': len(self.test_dataset)}
        }

        # Analyze caption statistics per split
        split_stats = {}
        for split_name, split_info in splits_info.items():
            dataset = split_info['dataset']
            all_captions = []

            # Collect all captions for this split
            for img in dataset.images:
                captions = dataset.image_to_captions.get(img, [])
                # Filter out NaN values and ensure all are strings
                clean_captions = [str(caption)
                                  for caption in captions if pd.notna(caption)]
                all_captions.extend(clean_captions)

            if all_captions:
                caption_lengths = [len(caption) for caption in all_captions]
                word_counts = [len(caption.split())
                               for caption in all_captions]

                split_stats[split_name] = {
                    'num_images': split_info['size'],
                    'num_captions': len(all_captions),
                    'avg_caption_length': np.mean(caption_lengths),
                    'avg_words_per_caption': np.mean(word_counts),
                    'captions_per_image': len(all_captions) / split_info['size'] if split_info['size'] > 0 else 0
                }

        # Log split statistics
        for split_name, stats in split_stats.items():
            self.wandb_logger.log_dataset_statistics(
                stats, prefix=f"splits_{split_name}")

        # Create split comparison visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Data Splits Analysis', fontsize=16, fontweight='bold')

        # Split sizes comparison
        split_names = list(split_stats.keys())
        split_sizes = [split_stats[name]['num_images'] for name in split_names]
        colors = ['#ff9999', '#66b3ff', '#99ff99']

        bars = ax1.bar(split_names, split_sizes, color=colors)
        ax1.set_title('Images per Split')
        ax1.set_ylabel('Number of Images')
        for bar, size in zip(bars, split_sizes):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(split_sizes)*0.01,
                     f'{size:,}', ha='center', va='bottom')

        # Captions per split
        caption_counts = [split_stats[name]['num_captions']
                          for name in split_names]
        bars = ax2.bar(split_names, caption_counts, color=colors)
        ax2.set_title('Captions per Split')
        ax2.set_ylabel('Number of Captions')
        for bar, count in zip(bars, caption_counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(caption_counts)*0.01,
                     f'{count:,}', ha='center', va='bottom')

        # Average caption length per split
        avg_lengths = [split_stats[name]['avg_caption_length']
                       for name in split_names]
        bars = ax3.bar(split_names, avg_lengths, color=colors)
        ax3.set_title('Average Caption Length per Split')
        ax3.set_ylabel('Average Length (Characters)')
        for bar, length in zip(bars, avg_lengths):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(avg_lengths)*0.01,
                     f'{length:.1f}', ha='center', va='bottom')

        # Captions per image ratio
        ratios = [split_stats[name]['captions_per_image']
                  for name in split_names]
        bars = ax4.bar(split_names, ratios, color=colors)
        ax4.set_title('Captions per Image Ratio')
        ax4.set_ylabel('Captions per Image')
        for bar, ratio in zip(bars, ratios):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(ratios)*0.01,
                     f'{ratio:.1f}', ha='center', va='bottom')

        plt.tight_layout()
        self.wandb_logger.save_and_log_figure(
            fig, 'data_splits_analysis')
        plt.close()

    def _create_sample_visualizations(self):
        """Create sample visualizations showing images with their captions."""
        self.logger.info("Creating sample visualizations...")

        # Create a grid of sample images with captions
        n_samples = 12
        n_cols = 4
        n_rows = 3

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 15))
        fig.suptitle('Sample Images with Captions',
                     fontsize=16, fontweight='bold')

        # Get random samples from training set
        sample_indices = np.random.choice(
            len(self.train_dataset), n_samples, replace=False)

        for i, idx in enumerate(sample_indices):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]

            try:
                # Get image and caption
                sample = self.train_dataset[idx]
                image = sample['image']
                captions = sample['caption']

                # Convert tensor to numpy for display
                if hasattr(image, 'numpy'):
                    # Denormalize the image
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    image_np = image.permute(1, 2, 0).numpy()
                    image_np = image_np * std + mean
                    image_np = np.clip(image_np, 0, 1)
                else:
                    image_np = np.array(image)

                ax.imshow(image_np)
                ax.axis('off')

                # Add caption as title (truncate if too long)
                caption_text = captions[0] if captions else "No caption"
                if len(caption_text) > 60:
                    caption_text = caption_text[:57] + "..."
                ax.set_title(caption_text, fontsize=8, wrap=True)

            except Exception as e:
                ax.text(0.5, 0.5, f'Error loading\nimage {idx}',
                        ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')

        plt.tight_layout()
        self.wandb_logger.save_and_log_figure(
            fig, 'sample_images')
        plt.close()

        self.logger.info("Sample visualizations created")

    def _generate_summary_report(self):
        """Generate a comprehensive summary report of the EDA findings."""
        self.logger.info("Generating summary report...")

        # Load basic statistics
        captions_df = pd.read_csv(
            self.config.dataset_caption_file_path, skipinitialspace=True)
        # Clean data - remove NaN values
        captions_df = captions_df.dropna(subset=['caption', 'image'])

        summary_stats = {
            'dataset_name': 'Flickr30K',
            'total_unique_images': captions_df['image'].nunique(),
            'total_captions': len(captions_df),
            'train_images': len(self.train_dataset),
            'val_images': len(self.val_dataset),
            'test_images': len(self.test_dataset),
            'avg_captions_per_image': len(captions_df) / captions_df['image'].nunique(),
            'config_train_size': self.config.train_size,
            'config_test_size': self.config.test_size,
            'config_val_size': 1.0 - self.config.train_size - self.config.test_size
        }

        # Create summary table
        summary_data = [
            ['Dataset', 'Flickr30K'],
            ['Total Unique Images',
                f"{summary_stats['total_unique_images']:,}"],
            ['Total Captions', f"{summary_stats['total_captions']:,}"],
            ['Avg Captions/Image',
                f"{summary_stats['avg_captions_per_image']:.1f}"],
            ['Train Images', f"{summary_stats['train_images']:,}"],
            ['Validation Images', f"{summary_stats['val_images']:,}"],
            ['Test Images', f"{summary_stats['test_images']:,}"],
            ['Train Ratio', f"{summary_stats['config_train_size']:.1%}"],
            ['Val Ratio', f"{summary_stats['config_val_size']:.1%}"],
            ['Test Ratio', f"{summary_stats['config_test_size']:.1%}"]
        ]

        # Log summary table to WandB
        self.wandb_logger.log_table(
            data=summary_data,
            columns=['Metric', 'Value'],
            name='dataset_summary'
        )

        # Log final summary metrics
        self.wandb_logger.log_dataset_statistics(
            summary_stats, prefix="summary")

        self.logger.info("Summary report generated and logged to WandB")

        # Log summary to console using logger
        self.logger.info("="*60, color=Colors.BLUE)
        self.logger.info("FLICKR30K DATASET EDA SUMMARY", color=Colors.BLUE)
        self.logger.info("="*60, color=Colors.BLUE)
        for metric, value in summary_data:
            self.logger.info(f"{metric:<25}: {value}", color=Colors.BLUE)
        self.logger.info("="*60, color=Colors.BLUE)
