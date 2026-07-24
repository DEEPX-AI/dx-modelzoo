from __future__ import annotations

import os
from glob import glob
from typing import Dict, List, Tuple

import cv2
import numpy as np

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.dataset import DATASET_REGISTRY

_INSTALL_GUIDE = """\
  [ImageNet ILSVRC2012] — Academic/Research use only

  1. Register at https://image-net.org/download-images.php
  2. Download "ILSVRC2012_img_val.tar" (validation images, ~6.3GB)
  3. Download "ILSVRC2012_devkit_t12.tar.gz" (devkit)
  4. Extract to: <DATA_ROOT>/ILSVRC2012/val/
     Expected structure:
       ILSVRC2012/val/
         n01440764/
           ILSVRC2012_val_00000293.JPEG
           ...
         n01443537/
           ...
  5. Use the devkit script to organize images into per-class subdirectories.

  License: Research/academic use only. Commercial use is NOT permitted.
  See: https://image-net.org/download-images.php
"""


@DATASET_REGISTRY.register
class ILSVRC2012(DatasetBase):
    """ImageNet dataset. Reads images from class subdirectories.

    Expected structure:
        data_dir/
            class_a/
                xxxxx.JPEG
            class_b/
                xxxxx.JPEG
    """

    def __init__(self, data_dir: str) -> None:
        self.ensure_exists(data_dir, _INSTALL_GUIDE)
        super().__init__(data_dir)
        self.image_files = sorted(glob(os.path.join(self.data_dir, "**", "*.JPEG"), recursive=True))
        self._class_map = self._build_class_map()
        self.class_list = self._build_class_list()

    def _build_class_map(self) -> Dict[str, int]:
        dirs = sorted([d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))])
        return {name: idx for idx, name in enumerate(dirs)}

    def _build_class_list(self) -> List[int]:
        class_list = []
        for path in self.image_files:
            class_name = path.split(os.sep)[-2]
            class_list.append(self._class_map[class_name])
        return class_list

    @property
    def class_map(self) -> Dict[str, int]:
        return self._class_map

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        img = cv2.imread(self.image_files[idx])
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {self.image_files[idx]}")
        img = self.preprocessing(img)
        label = self.class_list[idx]
        return img, label
