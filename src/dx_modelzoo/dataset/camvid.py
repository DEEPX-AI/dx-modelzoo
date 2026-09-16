from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.dataset import DATASET_REGISTRY

_INSTALL_GUIDE = """\
  [CamVid] — research/non-commercial use

  1. Download the CamVid images, annotations, and split lists from:
     http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/
     The SegNet CamVid preparation instructions are also useful:
     https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid
  2. Extract or arrange the dataset at: <DATA_ROOT>/camvid/
     Expected structure:
       camvid/
         images/
           ... image files ...
         annot/
           ... label files ...
         train.txt
         val.txt
         test.txt
  3. Each split file must contain one image/label pair per line, for example:
       images/0016E5_07959.png annot/0016E5_07959_L.png

  CamVid uses 11 semantic classes. Label IDs must be 0-10; 255 is ignored.
  Check the upstream dataset terms before redistribution or commercial use.
"""


@DATASET_REGISTRY.register
class CamVid(DatasetBase):
    """CamVid 11-class semantic segmentation dataset."""

    num_class = 11

    def __init__(self, data_dir: str, split: str = "val") -> None:
        self.ensure_exists(data_dir, _INSTALL_GUIDE)
        self.split = split
        super().__init__(data_dir)
        split_path = Path(data_dir, f"{self.split}.txt")
        if not split_path.is_file():
            raise FileNotFoundError(f"CamVid split not found: {split_path}")
        self.samples = []
        for line in split_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Invalid CamVid split line: {line}")
            image, label = (Path(data_dir, part) for part in parts)
            self.samples.append((image, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple:
        image_path, label_path = self.samples[index]
        image = cv2.imread(str(image_path))
        label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Failed to load image: {image_path}")
        if label is None:
            raise FileNotFoundError(f"Failed to load label: {label_path}")
        if label.shape != image.shape[:2]:
            label = cv2.resize(label, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        label = label.astype(np.int64, copy=False)
        invalid = (label >= self.num_class) & (label != 255)
        if np.any(invalid):
            raise ValueError(f"Unexpected CamVid labels in {label_path}")
        return self.preprocessing(image), label

    def validate(self) -> None:
        if not Path(self.data_dir, "images").is_dir() or not Path(self.data_dir, "annot").is_dir():
            raise RuntimeError(f"Expected images/ and annot/ directories under {self.data_dir}.")
        split_path = Path(self.data_dir, f"{self.split}.txt")
        if not split_path.is_file():
            raise RuntimeError(f"CamVid split not found: {split_path}")
        for line in split_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 2:
                raise RuntimeError(f"Invalid CamVid split line: {line}")
            image_path, label_path = (Path(self.data_dir, part) for part in parts)
            if not image_path.is_file() or not label_path.is_file():
                raise RuntimeError(f"CamVid split references missing files: {line}")
