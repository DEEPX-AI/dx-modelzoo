from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, List, Tuple

import cv2
import numpy as np

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.dataset import DATASET_REGISTRY

_INSTALL_GUIDE = """\
  [Oxford-IIIT Pet] — CC BY-SA 4.0 (free for research and commercial use)

  1. Download from https://www.robots.ox.ac.uk/~vgg/data/pets/
     - images.tar.gz (~800MB)
     - annotations.tar.gz
  2. Extract to: <DATA_ROOT>/Oxford-IIIT_Pet/
     Expected structure:
       Oxford-IIIT_Pet/
         images/
           Abyssinian_1.jpg
           ...
         annotations/
           test.txt
           trimaps/
             Abyssinian_1.png
             ...

  License: CC BY-SA 4.0 — free for research and commercial use.
  See: https://www.robots.ox.ac.uk/~vgg/data/pets/
"""


@DATASET_REGISTRY.register
class OxfordIIITPet(DatasetBase):
    num_class = 3

    def __init__(self, data_dir: str) -> None:
        self.ensure_exists(data_dir, _INSTALL_GUIDE)
        super().__init__(data_dir)
        self._label_preprocessing = None
        self.image_files, self.label_files = self._load_test_set()

    @property
    def label_preprocessing(self) -> Callable:
        if self._label_preprocessing is None:
            raise ValueError("Dataset's label preprocessing is not set.")
        return self._label_preprocessing

    @label_preprocessing.setter
    def label_preprocessing(self, value: Callable) -> None:
        self._label_preprocessing = value

    def _load_test_set(self) -> Tuple[List[str], List[str]]:
        test_txt = os.path.join(self.data_dir, "annotations", "test.txt")
        img_dir = os.path.join(self.data_dir, "images")
        trimap_dir = os.path.join(self.data_dir, "annotations", "trimaps")
        image_files, label_files = [], []
        with open(test_txt, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                name = line.split()[0]
                img_path = os.path.join(img_dir, name + ".jpg")
                label_path = os.path.join(trimap_dir, name + ".png")
                if os.path.exists(img_path) and os.path.exists(label_path):
                    image_files.append(img_path)
                    label_files.append(label_path)
        return image_files, label_files

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple:
        img = cv2.imread(self.image_files[idx])
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {self.image_files[idx]}")
        img = self.preprocessing(img)
        label = cv2.imread(self.label_files[idx], cv2.IMREAD_GRAYSCALE)
        if label is None:
            raise FileNotFoundError(f"Failed to load image: {self.label_files[idx]}")
        label = (label - 1).astype(np.int64)
        if self._label_preprocessing is not None:
            label = self.label_preprocessing(label)
        return img, label

    def validate(self) -> None:
        """Validate the dataset's integrity and structure.

        Raises:
            RuntimeError: If the dataset is invalid or corrupted.
        """
        # directory structure
        img_dir = os.path.join(self.data_dir, "images")
        trimap_dir = os.path.join(self.data_dir, "annotations", "trimaps")
        test_txt = os.path.join(self.data_dir, "annotations", "test.txt")
        is_correct_structure = (
            Path(self.data_dir).is_dir()
            and Path(img_dir).is_dir()
            and Path(trimap_dir).is_dir()
            and Path(test_txt).is_file()
        )
        if not is_correct_structure:
            raise RuntimeError(
                "Dataset directory structure is invalid. Expected 'images', 'annotations/trimaps' directories and"
                f" 'annotations/test.txt' file under {self.data_dir}."
            )

        # file existence
        NUM_DATA = 7390
        num_images = len(list(Path(img_dir).glob("*.jpg")))
        num_labels = len(list(Path(trimap_dir).glob("*.png")))
        if num_images != NUM_DATA:
            raise RuntimeError(f"Expected {NUM_DATA} images in {img_dir}, but found {num_images}.")
        if num_labels != NUM_DATA:
            raise RuntimeError(f"Expected {NUM_DATA} labels in {trimap_dir}, but found {num_labels}.")
