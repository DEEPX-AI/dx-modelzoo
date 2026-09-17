from __future__ import annotations

import os
from glob import glob
from typing import Tuple

import cv2
import numpy as np

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.dataset import DATASET_REGISTRY

_INSTALL_GUIDE = """\
  [LOL (Low-Light)] — Research use only

  1. Download from https://daooshee.github.io/BMVC2018website/
     or GitHub: https://github.com/weichen582/RetinexNet
  2. Download the LOL dataset (eval15 split).
  3. Extract to: <DATA_ROOT>/LOL/
     Expected structure:
       LOL/
         eval15/
           low/
             1.png
             ...
           high/
             1.png
             ...

  License: Research use only.
  See: https://daooshee.github.io/BMVC2018website/
"""


@DATASET_REGISTRY.register
class LOL(DatasetBase):
    def __init__(self, data_dir: str) -> None:
        self.ensure_exists(data_dir, _INSTALL_GUIDE)
        super().__init__(data_dir)
        eval_dir = os.path.join(data_dir, "eval15")
        self.low_files = sorted(glob(os.path.join(eval_dir, "low", "*.png")))
        self.high_files = sorted(glob(os.path.join(eval_dir, "high", "*.png")))
        if len(self.low_files) != len(self.high_files):
            raise ValueError(f"Mismatch: {len(self.low_files)} low vs {len(self.high_files)} high images")

    def __len__(self) -> int:
        return len(self.low_files)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        low_img = cv2.imread(self.low_files[idx])
        if low_img is None:
            raise FileNotFoundError(f"Failed to load image: {self.low_files[idx]}")
        high_img = cv2.imread(self.high_files[idx])
        if high_img is None:
            raise FileNotFoundError(f"Failed to load image: {self.high_files[idx]}")
        low_img = self.preprocessing(low_img)
        return low_img, high_img

    def validate(self) -> None:
        """Validate the dataset's integrity and structure.

        Raises:
            RuntimeError: If the dataset is invalid or corrupted.
        """
        # directory structure
        eval_dir = os.path.join(self.data_dir, "eval15")
        low_dir = os.path.join(eval_dir, "low")
        high_dir = os.path.join(eval_dir, "high")
        is_correct_structure = os.path.isdir(eval_dir) and os.path.isdir(low_dir) and os.path.isdir(high_dir)
        if not is_correct_structure:
            raise RuntimeError(
                "Dataset directory structure is invalid. Expected 'eval15/low' and 'eval15/high' directories under"
                f" {self.data_dir}."
            )

        # file existence
        NUM_DATA = 15
        num_low_images = len(glob(os.path.join(low_dir, "*.png")))
        num_high_images = len(glob(os.path.join(high_dir, "*.png")))
        if num_low_images != NUM_DATA:
            raise RuntimeError(f"Expected {NUM_DATA} low-light images in {low_dir}, but found {num_low_images}.")
        if num_high_images != NUM_DATA:
            raise RuntimeError(f"Expected {NUM_DATA} high-light images in {high_dir}, but found {num_high_images}.")
