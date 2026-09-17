from __future__ import annotations

import os
from copy import deepcopy
from glob import glob
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.dataset import DATASET_REGISTRY

_INSTALL_GUIDE = """\
  [DOTA v1] — Research use only

  1. Register and download from https://captain-whu.github.io/DOTA/dataset.html
  2. Download the validation split images and labels.
  3. Extract to: <DATA_ROOT>/DOTAv1/
     Expected structure:
       DOTAv1/
         images/
           val/
             P0003.png
             ...
         labels/
           val/
             P0003.txt
             ...

  License: Research use only. Requires registration.
  See: https://captain-whu.github.io/DOTA/dataset.html
"""

DOTA_CLASSES = [
    "plane",
    "baseball-diamond",
    "bridge",
    "ground-track-field",
    "small-vehicle",
    "large-vehicle",
    "ship",
    "tennis-court",
    "basketball-court",
    "storage-tank",
    "soccer-ball-field",
    "roundabout",
    "harbor",
    "swimming-pool",
    "helicopter",
]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(DOTA_CLASSES)}


@DATASET_REGISTRY.register
class DOTAv1(DatasetBase):
    def __init__(self, data_dir: str, split: str = "val") -> None:
        self.ensure_exists(data_dir, _INSTALL_GUIDE)
        super().__init__(data_dir)
        self.split = split
        self.image_dir = os.path.join(data_dir, "images", split)
        self.label_dir = os.path.join(data_dir, "labels", split)
        self.img_files = sorted(
            glob(os.path.join(self.image_dir, "*.jpg")) + glob(os.path.join(self.image_dir, "*.png"))
        )
        self.ids = [os.path.splitext(os.path.basename(p))[0] for p in self.img_files]
        valid = [
            i for i, img_id in enumerate(self.ids) if os.path.exists(os.path.join(self.label_dir, f"{img_id}.txt"))
        ]
        self.img_files = [self.img_files[i] for i in valid]
        self.ids = [self.ids[i] for i in valid]

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Tuple, str]:
        img = cv2.imread(self.img_files[idx])
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {self.img_files[idx]}")
        origin_img = deepcopy(img)
        img = self.preprocessing(img)
        return img, origin_img.shape, self.ids[idx]

    def validate(self) -> None:
        """Validate the dataset's integrity and structure.

        Raises:
            RuntimeError: If the dataset is invalid or corrupted.
        """
        image_dir = Path(self.data_dir, "images")
        label_dir = Path(self.data_dir, "labels")
        # directory structure
        is_correct_structure = Path(self.data_dir).is_dir() and image_dir.is_dir() and label_dir.is_dir()
        if not is_correct_structure:
            raise RuntimeError(
                "Dataset directory structure is invalid. Expected 'images' and 'labels' directories under"
                f" {self.data_dir}."
            )

        # file existence
        NUM_DATA = 2806
        NUM_LABELS = 3738
        num_images = len(list(image_dir.rglob("*.[jp][pn]g")))
        num_labels = len(list(label_dir.rglob("*.txt")))
        if num_images != NUM_DATA:
            raise RuntimeError(f"Expected {NUM_DATA} images in {image_dir}, but found {num_images}.")
        if num_labels != NUM_LABELS:
            raise RuntimeError(f"Expected {NUM_LABELS} labels in {label_dir}, but found {num_labels}.")
