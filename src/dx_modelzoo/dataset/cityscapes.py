from __future__ import annotations

import os
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
        with open(annpath, "r") as fr:
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
