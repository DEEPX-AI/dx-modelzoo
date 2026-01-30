import os
from copy import deepcopy
from glob import glob
from typing import Tuple

import cv2
import numpy as np

from dx_modelzoo.dataset import DatasetBase

# DOTA v1.0 classes
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

CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(DOTA_CLASSES)}


class DOTAV1Dataset(DatasetBase):
    """DOTA v1 dataset.

    Read images from pat and return as cv2.Image type.
    dataset_root's file tree should be like
        dataset_root/
            images/
                train/
                val/
                    P0003.jpg
                    ...
                test/
            labels/
                train/
                val/
                    P0003.txt
                    ...
                test/

    Args:
        data_dir (str): dataset root dir.
    """

    def __init__(self, data_dir: str, split: str = "val"):
        super().__init__(data_dir)
        self.split = split

        # Image and label directories
        self.image_dir = os.path.join(data_dir, "images", split)
        self.label_dir = os.path.join(data_dir, "labels", split)

        # Load image files
        self.img_files = sorted(
            glob(os.path.join(self.image_dir, "*.jpg")) + glob(os.path.join(self.image_dir, "*.png"))
        )
        self.ids = [os.path.splitext(os.path.basename(img_path))[0] for img_path in self.img_files]

        # Filter images that have corresponding label files
        valid_indices = []
        for idx, img_id in enumerate(self.ids):
            label_path = os.path.join(self.label_dir, f"{img_id}.txt")
            if os.path.exists(label_path):
                valid_indices.append(idx)

        self.img_files = [self.img_files[i] for i in valid_indices]
        self.ids = [self.ids[i] for i in valid_indices]

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Tuple[int, int, int], str]:
        """Get dataset item.

        Args:
            idx: Index of the item

        Returns:
            Tuple of (preprocessed_image, original_shape, image_id)
        """
        img = cv2.imread(self.img_files[idx])
        origin_img = deepcopy(img)
        img = self.preprocessing(img)
        return img, origin_img.shape, self.ids[idx]
