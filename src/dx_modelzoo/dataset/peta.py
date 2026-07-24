from __future__ import annotations

import os
from typing import List, Tuple

import cv2
import numpy as np

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.dataset import DATASET_REGISTRY

_INSTALL_GUIDE = """\
  [PETA (PEdesTrian Attribute)] — Research use only

  1. Download from https://mmlab.ie.cuhk.edu.hk/projects/PETA.html
     or contact the authors for the dataset.
  2. Required files:
     - PETA.mat (attribute annotations with splits)
     - images/ (pedestrian images, named 00001.png ~ XXXXX.png)
  3. Extract to: <DATA_ROOT>/PETA/
     Expected structure:
       PETA/
         PETA.mat
         images/
           00001.png
           00002.png
           ...

  Requires: scipy (for loading .mat files)
  License: Research use only.
  See: https://mmlab.ie.cuhk.edu.hk/projects/PETA.html
"""

PETA_ATTRIBUTES = [
    "personalLess30",
    "personalLess45",
    "personalLess60",
    "personalLarger60",
    "carryingBackpack",
    "carryingOther",
    "lowerBodyCasual",
    "upperBodyCasual",
    "lowerBodyFormal",
    "upperBodyFormal",
    "accessoryHat",
    "upperBodyJacket",
    "lowerBodyJeans",
    "footwearLeatherShoes",
    "upperBodyLogo",
    "hairLong",
    "personalMale",
    "carryingMessengerBag",
    "accessoryMuffler",
    "accessoryNothing",
    "carryingNothing",
    "upperBodyPlaid",
    "carryingPlasticBags",
    "footwearSandals",
    "footwearShoes",
    "lowerBodyShorts",
    "upperBodyShortSleeve",
    "lowerBodyShortSkirt",
    "footwearSneaker",
    "upperBodyThinStripes",
    "accessorySunglasses",
    "lowerBodyTrousers",
    "upperBodyTshirt",
    "upperBodyOther",
    "upperBodyVNeck",
]
NUM_ATTRIBUTES = 35


@DATASET_REGISTRY.register
class PETA(DatasetBase):
    def __init__(self, data_dir: str) -> None:
        self.ensure_exists(data_dir, _INSTALL_GUIDE)
        super().__init__(data_dir)
        self.samples: List[Tuple[str, np.ndarray]] = []
        self._load_annotations()

    def _load_annotations(self) -> None:
        try:
            import scipy.io
        except ImportError:
            raise ImportError("scipy is required for PETA dataset")
        mat_path = os.path.join(self.data_dir, "PETA.mat")
        mat = scipy.io.loadmat(mat_path)
        peta = mat["peta"][0, 0]
        data = peta[0]
        splits = peta[3]
        split0 = splits[0, 0][0, 0]
        test_indices = split0["test"].flatten()
        attr_cols = list(range(4, 4 + NUM_ATTRIBUTES))
        img_dir = os.path.join(self.data_dir, "images")
        for idx in test_indices:
            img_path = os.path.join(img_dir, f"{idx:05d}.png")
            if not os.path.exists(img_path):
                continue
            labels = data[idx - 1, attr_cols].astype(np.int64)
            self.samples.append((img_path, labels))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple:
        img_path, labels = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {img_path}")
        img = self.preprocessing(img)
        return img, labels, idx
