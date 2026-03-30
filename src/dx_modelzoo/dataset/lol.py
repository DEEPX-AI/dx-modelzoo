import os
from glob import glob
from typing import Tuple

import cv2
import numpy as np

from dx_modelzoo.dataset import DatasetBase


class LOLDataset(DatasetBase):
    """LOL (Low-Light) Dataset for low-light image enhancement evaluation.

    Loads paired low-light / normal-light images from eval15 split.

    Dataset structure:
        dataset_root/
            eval15/
                low/   (low-light input images)
                high/  (ground truth normal-light images)

    Args:
        data_dir: Path to dataset root directory (e.g., /mnt/datasets/LOL).
    """

    def __init__(self, data_dir: str):
        super().__init__(data_dir)
        eval_dir = os.path.join(data_dir, "eval15")
        low_dir = os.path.join(eval_dir, "low")
        high_dir = os.path.join(eval_dir, "high")

        self.low_files = sorted(glob(os.path.join(low_dir, "*.png")))
        self.high_files = sorted(glob(os.path.join(high_dir, "*.png")))

        if len(self.low_files) != len(self.high_files):
            raise ValueError(f"Mismatch: {len(self.low_files)} low vs {len(self.high_files)} high images")

    def __len__(self) -> int:
        return len(self.low_files)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get a paired sample.

        Returns:
            (low_image, high_image): Both BGR uint8 [H, W, 3].
            low_image is preprocessed by the model pipeline.
        """
        low_img = cv2.imread(self.low_files[idx])
        high_img = cv2.imread(self.high_files[idx])

        if low_img is None:
            raise ValueError(f"Failed to load: {self.low_files[idx]}")
        if high_img is None:
            raise ValueError(f"Failed to load: {self.high_files[idx]}")

        low_img = self.preprocessing(low_img)

        return low_img, high_img
