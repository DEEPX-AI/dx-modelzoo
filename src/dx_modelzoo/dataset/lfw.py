from __future__ import annotations

import csv
import os
from typing import List, Tuple

import cv2

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.dataset import DATASET_REGISTRY

_INSTALL_GUIDE = """\
  [LFW (Labeled Faces in the Wild)] — Research use only

  1. Download from http://vis-www.cs.umass.edu/lfw/
     - lfw-deepfunneled.tgz (deep-funneled aligned images)
     - pairs.txt (evaluation pairs)
  2. Convert pairs.txt to pairs.csv format.
  3. Extract to: <DATA_ROOT>/LFW/
     Expected structure:
       LFW/
         lfw-deepfunneled/
           Aaron_Eckhart/
             Aaron_Eckhart_0001.jpg
           ...
         pairs.csv

  License: Research use only.
  See: http://vis-www.cs.umass.edu/lfw/
"""


@DATASET_REGISTRY.register
class LFW(DatasetBase):
    def __init__(self, data_dir: str) -> None:
        self.ensure_exists(data_dir, _INSTALL_GUIDE)
        super().__init__(data_dir)
        self.image_dir = os.path.join(data_dir, "lfw-deepfunneled")
        self.pairs = self._load_pairs(os.path.join(data_dir, "pairs.csv"))

    @staticmethod
    def _load_pairs(pairs_csv: str) -> List[Tuple]:
        pairs = []
        with open(pairs_csv, newline="") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                cols = [c for c in row if c]
                if len(cols) == 3:
                    pairs.append((cols[0], int(cols[1]), cols[0], int(cols[2]), 1))
                elif len(cols) == 4:
                    pairs.append((cols[0], int(cols[1]), cols[2], int(cols[3]), 0))
        return pairs

    def _image_path(self, name: str, idx: int) -> str:
        return os.path.join(self.image_dir, name, f"{name}_{idx:04d}.jpg")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple:
        name1, i1, name2, i2, label = self.pairs[idx]
        img1 = cv2.imread(self._image_path(name1, i1))
        if img1 is None:
            raise FileNotFoundError(f"Failed to load image: {self._image_path(name1, i1)}")
        img2 = cv2.imread(self._image_path(name2, i2))
        if img2 is None:
            raise FileNotFoundError(f"Failed to load image: {self._image_path(name2, i2)}")
        img1 = self.preprocessing(img1)
        img2 = self.preprocessing(img2)
        return img1, img2, label
