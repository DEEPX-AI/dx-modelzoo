import os
from typing import List, Tuple

import cv2
import numpy as np
import scipy.io

from dx_modelzoo.dataset import DatasetBase

# First 35 PETA attributes (indices 0-34 of the full 105)
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


class PETADataset(DatasetBase):
    """PETA dataset for pedestrian attribute recognition.

    Loads test partition images with 35 selected binary attribute labels.

    Dataset structure:
        dataset_root/
            images/
                00001.png
                00002.png
                ...
            PETA.mat

    Args:
        data_dir (str): dataset root dir.
    """

    def __init__(self, data_dir: str):
        super().__init__(data_dir)
        self.samples: List[Tuple[str, np.ndarray]] = []
        self._load_annotations()

    def _load_annotations(self):
        mat_path = os.path.join(self.data_dir, "PETA.mat")
        mat = scipy.io.loadmat(mat_path)
        peta = mat["peta"][0, 0]

        data = peta[0]  # (19000, 109)
        # selected = peta[2][0]  # [5, 6, ..., 39]
        splits = peta[3]  # (5, 1) array of splits

        # Use split 0, test partition
        split0 = splits[0, 0][0, 0]
        test_indices = split0["test"].flatten()  # 1-based image indices

        # Use first NUM_ATTRIBUTES attributes (columns 4..4+NUM_ATTRIBUTES-1)
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

    def __getitem__(self, idx) -> Tuple:
        img_path, labels = self.samples[idx]
        img = cv2.imread(img_path)
        img = self.preprocessing(img)
        return img, labels, idx
