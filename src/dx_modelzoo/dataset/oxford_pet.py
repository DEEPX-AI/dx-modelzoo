import os
from typing import Callable, List, Tuple

import cv2
import numpy as np

from dx_modelzoo.dataset import DatasetBase


class OxfordPetDataset(DatasetBase):
    """Oxford-IIIT Pet dataset for image segmentation.

    Uses trimap annotations with 3 classes:
        0: Foreground (pet)
        1: Background
        2: Boundary

    Dataset structure:
        dataset_root/
            images/
                Name_N.jpg
            annotations/
                test.txt
                trimaps/
                    Name_N.png

    Args:
        data_dir (str): dataset root dir.
    """

    num_class = 3

    def __init__(self, data_dir: str):
        super().__init__(data_dir)
        self._label_preprocessing = None
        self.image_files, self.label_files = self._load_test_set()

    @property
    def label_preprocessing(self) -> Callable:
        if self._label_preprocessing is None:
            raise ValueError("Dataset's label PreProcessing is not set.")
        return self._label_preprocessing

    @label_preprocessing.setter
    def label_preprocessing(self, label_preprocessing: Callable) -> None:
        self._label_preprocessing = label_preprocessing

    def _load_test_set(self) -> Tuple[List[str], List[str]]:
        test_txt = os.path.join(self.data_dir, "annotations", "test.txt")
        img_dir = os.path.join(self.data_dir, "images")
        trimap_dir = os.path.join(self.data_dir, "annotations", "trimaps")

        image_files, label_files = [], []
        with open(test_txt, "r") as f:
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

    def __getitem__(self, idx) -> Tuple:
        img = cv2.imread(self.image_files[idx])
        img = self.preprocessing(img)

        label = cv2.imread(self.label_files[idx], cv2.IMREAD_GRAYSCALE)
        label = (label - 1).astype(np.int64)  # convert {1,2,3} -> {0,1,2}
        if self._label_preprocessing is not None:
            label = self.label_preprocessing(label)
        return img, label
