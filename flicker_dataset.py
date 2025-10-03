import os
from typing import Optional, Literal, Dict, List, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split

from config import AppConfig
from logger.app_logger import get_logger


class FlickerDataset(Dataset):
    """Dataset class for Flickr30k with train/val/test splits.
    
    Use create_splits() class method to efficiently create all three splits at once.
    """
    def __init__(
        self, 
        config: AppConfig, 
        split: Literal["train", "val", "test"] = "train",
        transform: Optional[transforms.Compose] = None
    ):
        self.config = config
        self.split = split
        self.logger = get_logger(config.app_name)
        
        # Load and validate captions
        self.captions = pd.read_csv(
            config.dataset_caption_file_path,
            skipinitialspace=True,
        )
        # Ensure columns exist
        if not {"image", "caption"}.issubset(self.captions.columns):
            raise ValueError(
                "Caption file must have columns: 'image', 'caption'")

        # Group captions by image
        grouped = self.captions.groupby(
            "image")["caption"].apply(list).reset_index()
        all_images = grouped["image"].tolist()
        all_image_to_captions = dict(
            zip(grouped["image"], grouped["caption"]))
        
        # Split the dataset
        self.images, self.image_to_captions = self._create_split(
            all_images, all_image_to_captions, split
        )
        
        self.logger.info(f"Created {split} split with {len(self.images)} images")
        self.__set_transform(transform)

    def __len__(self):
        # Number of unique images
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, any]:
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

    @staticmethod
    def collate_fn(batch: List[Dict[str, any]]) -> Dict[str, any]:
        """Custom collate function to handle multiple captions per image.

        Args:
            batch: List[Dict] with keys 'image' (Tensor CxHxW) and 'caption' (List[str])

        Returns:
            Dict with 'image' stacked into Tensor BxCxHxW and 'caption' as List[List[str]]
        """
        images = torch.stack([item["image"] for item in batch], dim=0)
        captions = [item["caption"] for item in batch]  # List[List[str]]
        return {"image": images, "caption": captions[0]}
        

    def _create_split(
        self, 
        all_images: List[str], 
        all_image_to_captions: Dict[str, List[str]], 
        split: Literal["train", "val", "test"]
    ) -> Tuple[List[str], Dict[str, List[str]]]:
        # Calculate split sizes
        train_size = self.config.train_size
        test_size = self.config.test_size
        val_size = 1.0 - train_size - test_size
        
        # First split: separate test set
        if test_size > 0:
            train_val_images, test_images = train_test_split(
                all_images, 
                test_size=test_size, 
                random_state=self.config.seed
            )
        else:
            train_val_images = all_images
            test_images = []
        
        # Second split: separate train and val from remaining data
        if val_size > 0:
            # Calculate val_size as proportion of remaining data
            val_size_adjusted = val_size / (train_size + val_size)
            train_images, val_images = train_test_split(
                train_val_images,
                test_size=val_size_adjusted,
                random_state=self.config.seed
            )
        else:
            train_images = train_val_images
            val_images = []
        
        # Select the appropriate split
        if split == "train":
            selected_images = train_images
        elif split == "val":
            selected_images = val_images
        elif split == "test":
            selected_images = test_images
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
        
        # Create filtered image_to_captions dict for this split
        selected_image_to_captions = {
            img: all_image_to_captions[img] for img in selected_images
        }
        
        return selected_images, selected_image_to_captions

    @classmethod
    def create_splits(
        cls, 
        config: AppConfig, 
        transforms_dict: Optional[Dict[str, transforms.Compose]] = None
    ) -> Tuple['FlickerDataset', 'FlickerDataset', 'FlickerDataset']:
        """Efficiently create train, val, and test datasets with shared preprocessing.
        
        Args:
            config: Application configuration
            transforms_dict: Optional dict with keys 'train', 'val', 'test' for different transforms
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # Load and process data once
        captions = pd.read_csv(
            config.dataset_caption_file_path,
            skipinitialspace=True,
        )
        if not {"image", "caption"}.issubset(captions.columns):
            raise ValueError(
                "Caption file must have columns: 'image', 'caption'")

        grouped = captions.groupby(
            "image")["caption"].apply(list).reset_index()
        all_images = grouped["image"].tolist()
        all_image_to_captions = dict(
            zip(grouped["image"], grouped["caption"]))
        
        # Do splits once
        train_size = config.train_size
        test_size = config.test_size
        val_size = 1.0 - train_size - test_size
        
        # First split: separate test set
        if test_size > 0:
            train_val_images, test_images = train_test_split(
                all_images, 
                test_size=test_size, 
                random_state=config.seed
            )
        else:
            train_val_images = all_images
            test_images = []
        
        # Second split: separate train and val from remaining data
        if val_size > 0:
            val_size_adjusted = val_size / (train_size + val_size)
            train_images, val_images = train_test_split(
                train_val_images,
                test_size=val_size_adjusted,
                random_state=config.seed
            )
        else:
            train_images = train_val_images
            val_images = []
        
        # Create datasets with pre-computed splits
        transforms_dict = transforms_dict or {}
        
        train_dataset = cls._create_from_split(
            config, train_images, all_image_to_captions, 
            "train", transforms_dict.get("train")
        )
        val_dataset = cls._create_from_split(
            config, val_images, all_image_to_captions, 
            "val", transforms_dict.get("val")
        )
        test_dataset = cls._create_from_split(
            config, test_images, all_image_to_captions, 
            "test", transforms_dict.get("test")
        )
        
        return train_dataset, val_dataset, test_dataset
    
    @classmethod
    def _create_from_split(
        cls,
        config: AppConfig,
        images: List[str],
        all_image_to_captions: Dict[str, List[str]],
        split: Literal["train", "val", "test"],
        transform: Optional[transforms.Compose] = None
    ) -> 'FlickerDataset':
        """Create dataset instance from pre-computed split."""
        instance = cls.__new__(cls)  # Create without calling __init__
        instance.config = config
        instance.split = split
        instance.logger = get_logger(config.app_name)
        instance.images = images
        instance.image_to_captions = {
            img: all_image_to_captions[img] for img in images
        }
        instance.logger.info(f"Created {split} split with {len(images)} images")
        instance.__set_transform(transform)
        return instance

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
