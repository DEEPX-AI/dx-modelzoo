from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.dataset import DATASET_REGISTRY

_INSTALL_GUIDE = """\
  [Cityscapes] — Academic/Research use only

  1. Register at https://www.cityscapes-dataset.com/register/
  2. Download:
     - leftImg8bit_trainvaltest.zip (11GB)
     - gtFine_trainvaltest.zip (241MB)
  3. Extract to: <DATA_ROOT>/cityscapes/
     Expected structure:
       cityscapes/
         leftImg8bit/
           val/
             frankfurt/
               frankfurt_000000_000294_leftImg8bit.png
               ...
         gtFine/
           val/
             frankfurt/
               frankfurt_000000_000294_gtFine_labelIds.png
               ...
         val.txt  (image,label pairs)

  License: Academic/Research use only. Requires registration.
  See: https://www.cityscapes-dataset.com/license/
"""

CITYSCAPES_LABEL_MAP = np.full(256, 255, dtype=np.uint8)
_TRAIN_IDS = {
    7: 0,
    8: 1,
    11: 2,
    12: 3,
    13: 4,
    17: 5,
    19: 6,
    20: 7,
    21: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    31: 16,
    32: 17,
    33: 18,
}
for _id, _tid in _TRAIN_IDS.items():
    CITYSCAPES_LABEL_MAP[_id] = _tid


@DATASET_REGISTRY.register
class Cityscapes(DatasetBase):
    num_class = 19

    def __init__(self, data_dir: str) -> None:
        self.ensure_exists(data_dir, _INSTALL_GUIDE)
        super().__init__(data_dir)
        annpath = os.path.join(self.data_dir, "val.txt")
        with open(annpath, "r", encoding="utf-8") as fr:
            pairs = fr.read().splitlines()
        self.img_paths, self.lb_paths = [], []
        for pair in pairs:
            imgpth, lbpth = pair.split(",")
            self.img_paths.append(os.path.join(self.data_dir, imgpth))
            self.lb_paths.append(os.path.join(self.data_dir, lbpth))
        self.lb_map = CITYSCAPES_LABEL_MAP

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, index: int) -> Tuple:
        img = cv2.imread(self.img_paths[index])
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {self.img_paths[index]}")
        label = cv2.imread(self.lb_paths[index], 0)
        if label is None:
            raise FileNotFoundError(f"Failed to load image: {self.lb_paths[index]}")
        if self.lb_map is not None:
            label = self.lb_map[label]
        img = self.preprocessing(img)
        return img, label

    def validate(self) -> None:
        """Validate the dataset's integrity and structure.

        Raises:
            RuntimeError: If the dataset is invalid or corrupted.
        """
        # directory structure
        is_correct_structure = (
            Path(self.data_dir).is_dir()
            and Path(self.data_dir, "leftImg8bit").is_dir()
            and Path(self.data_dir, "gtFine").is_dir()
            and Path(self.data_dir, "val.txt").is_file()
        )
        if not is_correct_structure:
            raise RuntimeError(
                "Dataset directory structure is invalid. Expected 'leftImg8bit', 'gtFine' directories and "
                f"'val.txt' file under {self.data_dir}."
            )

        # file existence
        NUM_DATA = 500
        num_images = len(list(Path(self.data_dir, "leftImg8bit", "val").glob("*/*.png")))
        num_labels = len(list(Path(self.data_dir, "gtFine", "val").glob("*/*_labelIds.png")))
        if num_images != NUM_DATA:
            raise RuntimeError(
                f"Expected {NUM_DATA} images in {self.data_dir}/leftImg8bit/val, but found {num_images}."
            )
        if num_labels != NUM_DATA:
            raise RuntimeError(f"Expected {NUM_DATA} labels in {self.data_dir}/gtFine/val, but found {num_labels}.")
