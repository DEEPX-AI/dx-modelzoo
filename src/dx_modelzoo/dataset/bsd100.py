import os
from glob import glob
from typing import Tuple

import cv2
import numpy as np

from dx_modelzoo.dataset import DatasetBase


class BSD100Dataset(DatasetBase):
    """BSD100 Dataset Class for Super Resolution.

    This dataset loads image pairs for super resolution evaluation.
    All image preprocessing (Y channel conversion, resizing) is handled by the model's preprocessing pipeline.

    Dataset structure should be:
        dataset_root/
            HR/  # High Resolution images
                xxxxx.png
                xxxxx.jpg
                ...
            LR/  # Low Resolution images (optional, can be generated on-the-fly)
                xxxxx.png
                xxxxx.jpg
                ...

    If LR directory doesn't exist, HR images will be used for both LR and HR (LR generation handled by preprocessing).

    Args:
        data_dir (str): Path to dataset root directory
    """

    def __init__(self, data_dir: str):
        super().__init__(data_dir)
        self.set_lr_hr_files(data_dir)

    def set_lr_hr_files(self, data_dir):
        self._data_dir = data_dir
        hr_dir = os.path.join(data_dir, "HR")
        if os.path.exists(hr_dir):
            self.hr_files = sorted(glob(os.path.join(hr_dir, "*")))
        else:
            self.hr_files = sorted(glob(os.path.join(data_dir, "*")))

        lr_dir = os.path.join(data_dir, "LR")
        if os.path.exists(lr_dir):
            self.lr_files = sorted(glob(os.path.join(lr_dir, "*")))
            if len(self.lr_files) != len(self.hr_files):
                raise ValueError(
                    f"HR and LR directories must contain same number of files. "
                    f"HR: {len(self.hr_files)}, LR: {len(self.lr_files)}"
                )
        else:
            self.lr_files = None

    @property
    def data_dir(self) -> int:
        if self._data_dir is None:
            raise ValueError("data_dir property is not set.")
        return self._data_dir

    @data_dir.setter
    def data_dir(self, data_dir):
        self.set_lr_hr_files(data_dir)

    def __len__(self) -> int:
        return len(self.hr_files)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get a data sample.

        Args:
            idx (int): Sample index

        Returns:
            Tuple[np.ndarray, np.ndarray]: (lr_image, hr_image) both as BGR images in [0, 255] range
        """
        hr_image = cv2.imread(self.hr_files[idx])
        if hr_image is None:
            raise ValueError(f"Failed to load image: {self.hr_files[idx]}")

        if self.lr_files is not None:
            lr_image = cv2.imread(self.lr_files[idx])
            if lr_image is None:
                raise ValueError(f"Failed to load LR image: {self.lr_files[idx]}")
        else:
            lr_image = hr_image.copy()

        if hasattr(self, "lr_preprocessing") and self.lr_preprocessing is not None:
            lr_image = cv2.cvtColor(hr_image, cv2.COLOR_RGB2YCrCb)
            lr_image = self.lr_preprocessing(lr_image)
        if hasattr(self, "hr_preprocessing") and self.hr_preprocessing is not None:
            hr_image = cv2.cvtColor(hr_image, cv2.COLOR_RGB2YCrCb)
            hr_image = self.hr_preprocessing(hr_image)

        return lr_image, hr_image

    def set_lr_preprocessing(self, lr_preprocessing) -> None:
        """Set LR-specific preprocessing pipeline."""
        self.lr_preprocessing = lr_preprocessing

    def set_hr_preprocessing(self, hr_preprocessing) -> None:
        """Set HR-specific preprocessing pipeline."""
        self.hr_preprocessing = hr_preprocessing
