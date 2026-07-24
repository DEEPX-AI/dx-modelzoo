from __future__ import annotations

import os
from glob import glob
from typing import Tuple

import cv2
import numpy as np

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.dataset import DATASET_REGISTRY

_INSTALL_GUIDE = """\
  [ADE20K — SceneParse150 / ADEChallengeData2016] — Research use

  1. Download ADEChallengeData2016.zip from
     http://sceneparsing.csail.mit.edu/ or the MIT SceneParse150 page.
  2. Extract to: <DATA_ROOT>/ade20k-sceneparse150/
     Expected structure:
       ade20k-sceneparse150/
         images/
           validation/ADE_val_00000001.jpg ...
         annotations/
           validation/ADE_val_00000001.png ...   (indexed PNG, values 0-150)
         objectInfo150.txt

  Label encoding: 0 = unlabeled (ignored), 1-150 = the 150 SceneParse150
  classes. This dataset maps them to 0-149 (model output index) with 0 -> 255
  (ignore).
"""

# reduce_zero_label: 0 (unlabeled) -> 255 (ignore); 1..150 -> 0..149
ADE20K_LABEL_MAP = np.full(256, 255, dtype=np.uint8)
ADE20K_LABEL_MAP[1:151] = np.arange(150, dtype=np.uint8)


@DATASET_REGISTRY.register
class ADE20K(DatasetBase):
    """ADE20K SceneParse150 semantic segmentation (150 classes).

    Evaluation is performed at a fixed ``eval_size`` (default 512x512): the
    image is resized by the preprocessing pipeline and the GT label is resized
    here with nearest interpolation so prediction and label share the same
    spatial resolution.
    """

    num_class = 150

    def __init__(self, data_dir: str, eval_size: int = 512) -> None:
        self.ensure_exists(data_dir, _INSTALL_GUIDE)
        super().__init__(data_dir)
        self.eval_size = int(eval_size)
        self.img_paths = sorted(glob(os.path.join(data_dir, "images", "validation", "*.jpg")))
        if not self.img_paths:
            raise FileNotFoundError(f"No validation images found under {data_dir}/images/validation")
        self.lb_paths = [
            os.path.join(data_dir, "annotations", "validation", os.path.splitext(os.path.basename(p))[0] + ".png")
            for p in self.img_paths
        ]
        self.lb_map = ADE20K_LABEL_MAP

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, index: int) -> Tuple:
        img = cv2.imread(self.img_paths[index])
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {self.img_paths[index]}")
        label = cv2.imread(self.lb_paths[index], cv2.IMREAD_GRAYSCALE)
        if label is None:
            raise FileNotFoundError(f"Failed to load label: {self.lb_paths[index]}")
        label = self.lb_map[label]
        label = cv2.resize(
            label, (self.eval_size, self.eval_size), interpolation=cv2.INTER_NEAREST
        )
        img = self.preprocessing(img)
        return img, label
