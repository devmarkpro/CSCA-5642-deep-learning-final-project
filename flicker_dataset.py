import io
from pathlib import Path
from typing import Optional, Literal

import os
import csv

import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
import torch

from config import AppConfig
from logger.app_logger import get_logger


class Flicker30kDataset(Dataset):
    def __init__(
        self, config: AppConfig, transform: Optional[transforms.Compose] = None
    ):
        self.config = config
        self.logger = get_logger(config.app_name)
        # Expecting a CSV with columns: 'image', 'caption'
        # Use skipinitialspace so values after commas don't keep a leading space.
        self.captions = pd.read_csv(
            config.dataset_caption_file_path,
            skipinitialspace=True,
        )
        # Ensure columns exist
        if not {"image", "caption"}.issubset(self.captions.columns):
            raise ValueError("Caption file must have columns: 'image', 'caption'")

        # Normalize and clean strings: collapse spaces, strip quotes/spaces
        self.captions["image"] = (
            self.captions["image"]
            .astype(str)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip(" \"'")
        )
        self.captions["caption"] = (
            self.captions["caption"]
            .astype(str)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip(" \"'")
        )

        grouped = self.captions.groupby("image")["caption"].apply(list).reset_index()
        self.images = grouped["image"].tolist()
        self.image_to_captions = dict(zip(grouped["image"], grouped["caption"]))

        self.__set_transform(transform)

    def __len__(self):
        # Number of unique images
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_file = self.images[idx]
        img_path = os.path.join(self.config.dataset_image_dir, img_file)
        image = Image.open(img_path).convert("RGB")
        captions = list(self.image_to_captions.get(img_file, []))

        if self.transform:
            image = self.transform(image)
        sample = {
            "image": image,
            "caption": captions,
        }
        return sample

    def __set_transform(self, transform: Optional[transforms.Compose]) -> None:
        self.transform = (
            transform
            if transform is not None
            else transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        )

    @staticmethod
    def collate_fn(batch):
        """Custom collate function to handle multiple captions per image.

        Args:
            batch: List[Dict] with keys 'image' (Tensor CxHxW) and 'caption' (List[str])

        Returns:
            Dict with 'image' stacked into Tensor BxCxHxW and 'caption' as List[List[str]]
        """
        images = torch.stack([item["image"] for item in batch], dim=0)
        captions = [item["caption"] for item in batch]  # List[List[str]]
        return {"image": images, "caption": captions[0]}
