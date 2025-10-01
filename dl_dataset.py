from pathlib import Path
from typing import Optional, Literal

import os
import csv
from torchvision.datasets import Flickr30k

from torchvision import transforms
import torch

from config import AppConfig
from logger.app_logger import get_logger


class Dataset:
    def __init__(
        self, config: AppConfig, transform: Optional[transforms.Compose] = None
    ):
        self.config = config
        self.logger = get_logger(config.app_name)
        self.__set_transform(transform)
        self.__load_dataset()

    def dataloader(
        self, dataset: Optional[Literal["train", "test", "val"]] = "train"
    ) -> torch.utils.data.DataLoader:
        """
        Return dataloader for the specified dataset. by default returns train dataloader
        """
        if dataset is None or dataset == "train":
            return torch.utils.data.DataLoader(
                self.train_ds,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
            )
        elif dataset == "test":
            return torch.utils.data.DataLoader(
                self.test_ds,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
            )
        elif dataset == "val":
            return torch.utils.data.DataLoader(
                self.val_ds,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
            )
        raise ValueError(f"Unknown dataset: {dataset}")

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

    def __load_dataset(self) -> None:
        self.logger.debug("Loading dataset")
        # Ensure image root exists (helps catch case-sensitive path issues like 'Images' vs 'images')
        if not os.path.exists(self.config.dataset_image_dir):
            self.logger.warning(
                f"dataset_image_dir does not exist: {self.config.dataset_image_dir}. "
                "Verify the path matches your data location (case-sensitive)."
            )

        # Flickr30k expects a TSV file (img_id \t caption per line, no header)
        ann_to_use = self.extract_captions()

        ds = Flickr30k(
            root=self.config.dataset_image_dir,
            ann_file=ann_to_use,
            transform=self.transform,
        )

        total_size = len(ds)
        train_size = int(self.config.train_size * total_size)
        test_size = int(self.config.test_size * total_size)
        val_size = total_size - train_size - test_size
        self.train_ds, self.test_ds, self.val_ds = torch.utils.data.random_split(
            ds, [train_size, test_size, val_size]
        )
        self.logger.info(
            f"Loaded dataset with {total_size} images, {train_size} train, {test_size} test, {val_size} val"
        )

    def extract_captions(self) -> str:
        """
        Extract captions from the caption file. If the caption file is not a TSV file, convert it to a TSV file.
        """
        ann_path = self.config.dataset_caption_file_path
        ann_to_use = ann_path

        if Path(ann_to_use).suffix != ".tsv":
            self.logger.warning(
                f"Caption file is not a CSV: {ann_to_use}. "
                "Flickr30k expects a TSV file (img_id\tcaption per line, no header)."
                "Checking for TSV file..."
            )
            # replace .csv with .tsv and check if tsv file exists, instead of csv
            tsv_path = ann_path.replace(".csv", ".tsv")
            if Path(tsv_path).exists():
                ann_to_use = tsv_path
                self.logger.info(f"Found TSV file: {ann_to_use}, using it.")
            else:
                self.logger.warning(
                    f"could not find TSV file: {tsv_path}, converting csv to tsv..."
                )
                ann_to_use = self.__convert_csv_to_tsv(
                    self.config.dataset_caption_file_path
                )
                self.logger.info(
                    f"Wrote TSV captions to: {ann_to_use}, use this file for Flickr30k dataset to avoid conversion next time."
                )
        return ann_to_use

    def __convert_csv_to_tsv(self, ann_path: str) -> str:
        try:
            with open(ann_path, "r", encoding="utf-8") as f:
                first_line = f.readline()
                self.logger.info(
                    "Detected CSV captions; converting to TSV for Flickr30k compatibility."
                )
                # Convert CSV -> TSV (image, caption)
                tsv_path = self.config.dataset_caption_file_path.replace(".csv", ".tsv")
                # Re-open to iterate from start
                f.seek(0)
                reader = csv.reader(f)
                header = next(reader, None)
                # If header present and looks like [image, caption], skip it
                has_header = False
                if (
                    header
                    and len(header) >= 2
                    and (
                        header[0].strip().lower() == "image"
                        or header[0].strip().lower() == "img"
                    )
                ):
                    has_header = True
                # If first line wasn't header, include it back
                rows_iter = reader if has_header else ([header] if header else [])
                # Write TSV
                with open(tsv_path, "w", encoding="utf-8") as out_f:
                    for row in rows_iter:
                        if not row:
                            continue
                        img_id = (row[0] or "").strip()
                        caption = ",".join(row[1:]).strip() if len(row) > 1 else ""
                        if not img_id or not caption:
                            continue
                        out_f.write(f"{img_id}\t{caption}\n")
                ann_to_use = tsv_path
                self.logger.info(f"Wrote TSV captions to: {ann_to_use}")
                return tsv_path

        except FileNotFoundError:
            self.logger.error(
                f"Caption file not found at: {ann_path}. Check AppConfig.dataset_caption_file_path."
            )
            raise
