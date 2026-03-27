import csv
import os
from typing import List, Tuple

import cv2
import numpy as np

from dx_modelzoo.dataset import DatasetBase

# CelebA 40 binary attributes
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


class CelebADataset(DatasetBase):
    """CelebA dataset for face attribute recognition.

    Loads test partition images with 40 binary attribute labels.

    Dataset structure:
        dataset_root/
            img_align_celeba/
                000001.jpg ...
            list_attr_celeba.csv
            list_eval_partition.csv

    Args:
        data_dir (str): dataset root dir.
    """

    NUM_ATTRIBUTES = 40

    def __init__(self, data_dir: str):
        super().__init__(data_dir)
        self.samples: List[Tuple[str, np.ndarray]] = []
        self._load_annotations()

    def _load_annotations(self):
        """Load test partition images and their attribute labels."""
        partition_path = os.path.join(self.data_dir, "list_eval_partition.csv")
        attr_path = os.path.join(self.data_dir, "list_attr_celeba.csv")
        img_dir = os.path.join(self.data_dir, "img_align_celeba")

        # Read test partition image IDs (partition == 2)
        test_ids = set()
        with open(partition_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["partition"] == "2":
                    test_ids.add(row["image_id"])

        # Read attributes for test images
        with open(attr_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_id = row["image_id"]
                if img_id not in test_ids:
                    continue

                img_path = os.path.join(img_dir, img_id)
                if not os.path.exists(img_path):
                    continue

                # Convert -1/1 labels to 0/1
                labels = np.array(
                    [max(0, int(row[attr])) for attr in CELEBA_ATTRIBUTES],
                    dtype=np.int64,
                )
                self.samples.append((img_path, labels))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple:
        img_path, labels = self.samples[idx]
        img = cv2.imread(img_path)
        img = self.preprocessing(img)
        return img, labels, idx
