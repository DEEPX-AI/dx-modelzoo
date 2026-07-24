from __future__ import annotations

import csv
import os
from typing import List, Tuple

import cv2
import numpy as np

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.dataset import DATASET_REGISTRY

_INSTALL_GUIDE = """\
  [CelebA] — Non-commercial research only

  1. Download from https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
     or Kaggle: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
  2. Required files:
     - img_align_celeba.zip (aligned & cropped images)
     - list_attr_celeba.csv (attribute annotations)
     - list_eval_partition.csv (train/val/test split)
  3. Extract to: <DATA_ROOT>/CelebA/
     Expected structure:
       CelebA/
         img_align_celeba/
           000001.jpg
           000002.jpg
           ...
         list_attr_celeba.csv
         list_eval_partition.csv

  License: Non-commercial research only.
  See: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
"""

CELEBA_ATTRIBUTES = [
    "5_o_Clock_Shadow",
    "Arched_Eyebrows",
    "Attractive",
    "Bags_Under_Eyes",
    "Bald",
    "Bangs",
    "Big_Lips",
    "Big_Nose",
    "Black_Hair",
    "Blond_Hair",
    "Blurry",
    "Brown_Hair",
    "Bushy_Eyebrows",
    "Chubby",
    "Double_Chin",
    "Eyeglasses",
    "Goatee",
    "Gray_Hair",
    "Heavy_Makeup",
    "High_Cheekbones",
    "Male",
    "Mouth_Slightly_Open",
    "Mustache",
    "Narrow_Eyes",
    "No_Beard",
    "Oval_Face",
    "Pale_Skin",
    "Pointy_Nose",
    "Receding_Hairline",
    "Rosy_Cheeks",
    "Sideburns",
    "Smiling",
    "Straight_Hair",
    "Wavy_Hair",
    "Wearing_Earrings",
    "Wearing_Hat",
    "Wearing_Lipstick",
    "Wearing_Necklace",
    "Wearing_Necktie",
    "Young",
]


@DATASET_REGISTRY.register
class CelebA(DatasetBase):
    NUM_ATTRIBUTES = 40

    def __init__(self, data_dir: str) -> None:
        self.ensure_exists(data_dir, _INSTALL_GUIDE)
        super().__init__(data_dir)
        self.samples: List[Tuple[str, np.ndarray]] = []
        self._load_annotations()

    def _load_annotations(self) -> None:
        partition_path = os.path.join(self.data_dir, "list_eval_partition.csv")
        attr_path = os.path.join(self.data_dir, "list_attr_celeba.csv")
        img_dir = os.path.join(self.data_dir, "img_align_celeba")
        test_ids = set()
        with open(partition_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["partition"] == "2":
                    test_ids.add(row["image_id"])
        with open(attr_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_id = row["image_id"]
                if img_id not in test_ids:
                    continue
                img_path = os.path.join(img_dir, img_id)
                if not os.path.exists(img_path):
                    continue
                labels = np.array([max(0, int(row[attr])) for attr in CELEBA_ATTRIBUTES], dtype=np.int64)
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
