from __future__ import annotations

import os
from glob import glob
from typing import Tuple

import cv2
import numpy as np

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.dataset import DATASET_REGISTRY

_INSTALL_GUIDE = """\
  [BSD68] — BSD license (free for research and commercial use)

  1. Download from https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/
     or: https://github.com/clausmichele/CBSD68-dataset
  2. Extract the 68 test images.
  3. Place in: <DATA_ROOT>/BSD68/
     Expected structure:
       BSD68/
         test001.png
         test002.png
         ... (68 images)

       BSD100/                         (super-resolution; one dir per scale)
         bicubic_2x/
           HR/
             101085.png
             ... (100 images)
           LR/
             101085.png
             ...
         bicubic_3x/  HR/ LR/
         bicubic_4x/  HR/ LR/
         bicubic_8x/  HR/ LR/
       NOTE: the BSD100 dataset reads ``HR``/``LR`` directly under its
       ``eval_path``, so point ``eval_path`` at a scale dir
       (e.g. ``BSD100/bicubic_4x``), not at ``BSD100`` itself.

       CBSD68/
         0000.png
         0001.png
         ... (68 color images)

  License: BSD — free for research and commercial use.
"""


@DATASET_REGISTRY.register
class BSD68(DatasetBase):
    def __init__(self, data_dir: str) -> None:
        self.ensure_exists(data_dir, _INSTALL_GUIDE)
        super().__init__(data_dir)
        self.img_files = sorted(glob(os.path.join(self.data_dir, "*")))

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, idx: int) -> Tuple:
        origin_img = cv2.imread(self.img_files[idx])
        if origin_img is None:
            raise FileNotFoundError(f"Failed to load image: {self.img_files[idx]}")
        inp = self.preprocessing(origin_img)
        if inp.ndim == 3 and inp.shape[-1] in (1, 3):
            h, w = inp.shape[0], inp.shape[1]
        else:
            h, w = inp.shape[-2], inp.shape[-1]
        label = cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY)
        if label.shape[0] != h or label.shape[1] != w:
            label = cv2.resize(label, (w, h), interpolation=cv2.INTER_AREA)
        return inp, label


@DATASET_REGISTRY.register
class BSD100(DatasetBase):
    def __init__(self, data_dir: str) -> None:
        self.ensure_exists(data_dir, _INSTALL_GUIDE)
        super().__init__(data_dir)
        self.set_lr_hr_files(data_dir)

    def set_lr_hr_files(self, data_dir: str) -> None:
        self._data_dir = data_dir
        hr_dir = os.path.join(data_dir, "HR")
        self.hr_files = (
            sorted(glob(os.path.join(hr_dir, "*")))
            if os.path.exists(hr_dir)
            else sorted(glob(os.path.join(data_dir, "*")))
        )
        lr_dir = os.path.join(data_dir, "LR")
        if os.path.exists(lr_dir):
            self.lr_files = sorted(glob(os.path.join(lr_dir, "*")))
        else:
            self.lr_files = None

    @property
    def data_dir(self) -> str:
        return self._data_dir

    @data_dir.setter
    def data_dir(self, data_dir: str) -> None:
        self.set_lr_hr_files(data_dir)

    def __len__(self) -> int:
        return len(self.hr_files)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        hr_image = cv2.imread(self.hr_files[idx])
        if hr_image is None:
            raise FileNotFoundError(f"Failed to load image: {self.hr_files[idx]}")
        lr_image = cv2.imread(self.lr_files[idx]) if self.lr_files else hr_image.copy()
        if self.lr_files and lr_image is None:
            raise FileNotFoundError(f"Failed to load image: {self.lr_files[idx]}")
        if hasattr(self, "lr_preprocessing") and self.lr_preprocessing is not None:
            lr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2YCrCb)
            lr_image = self.lr_preprocessing(lr_image)
        elif self._preprocessing is not None:
            lr_image = self.preprocessing(lr_image)
        if hasattr(self, "hr_preprocessing") and self.hr_preprocessing is not None:
            hr_image = self.hr_preprocessing(hr_image)
        return lr_image, hr_image

    def set_lr_preprocessing(self, lr_preprocessing) -> None:
        self.lr_preprocessing = lr_preprocessing

    def set_hr_preprocessing(self, hr_preprocessing) -> None:
        self.hr_preprocessing = hr_preprocessing


@DATASET_REGISTRY.register
class CBSD68(DatasetBase):
    def __init__(self, data_dir: str) -> None:
        self.ensure_exists(data_dir, _INSTALL_GUIDE)
        super().__init__(data_dir)
        self.img_files = sorted(glob(os.path.join(self.data_dir, "*")))

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, idx: int) -> Tuple:
        origin_img = cv2.imread(self.img_files[idx])
        if origin_img is None:
            raise FileNotFoundError(f"Failed to load image: {self.img_files[idx]}")
        inp = self.preprocessing(origin_img)
        if inp.ndim == 3 and inp.shape[-1] in (1, 3):
            h, w = inp.shape[0], inp.shape[1]
        else:
            h, w = inp.shape[-2], inp.shape[-1]
        label = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)
        if label.shape[0] != h or label.shape[1] != w:
            label = cv2.resize(label, (w, h), interpolation=cv2.INTER_AREA)
        return inp, label
