import csv
import os

import cv2

from dx_modelzoo.dataset import DatasetBase


class LFWDataset(DatasetBase):
    """LFW (Labeled Faces in the Wild) Dataset for face verification.

    Loads image pairs from ``pairs.csv`` under *data_dir*.

    Each sample is a tuple ``(img1, img2, label)`` where *label* is
    ``1`` for a genuine (same-person) pair and ``0`` for an impostor pair.

    Expected directory layout::

        data_dir/
            pairs.csv
            lfw-deepfunneled/
                Person_Name/
                    Person_Name_0001.jpg
                    ...

    Args:
        data_dir: Root directory that contains ``pairs.csv`` and
            ``lfw-deepfunneled/``.
    """

    def __init__(self, data_dir: str):
        super().__init__(data_dir)
        self.image_dir = os.path.join(data_dir, "lfw-deepfunneled")
        self.pairs = self._load_pairs(os.path.join(data_dir, "pairs.csv"))

    # ------------------------------------------------------------------
    # Pair loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_pairs(pairs_csv: str):
        """Parse ``pairs.csv`` into a list of ``(path1, path2, label)``."""
        pairs = []
        with open(pairs_csv, newline="") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                cols = [c for c in row if c]
                if len(cols) == 3:
                    # positive pair: name, idx1, idx2
                    name, i1, i2 = cols[0], int(cols[1]), int(cols[2])
                    pairs.append((name, i1, name, i2, 1))
                elif len(cols) == 4:
                    # negative pair: name1, idx1, name2, idx2
                    pairs.append((cols[0], int(cols[1]), cols[2], int(cols[3]), 0))
        return pairs

    def _image_path(self, name: str, idx: int) -> str:
        return os.path.join(self.image_dir, name, f"{name}_{idx:04d}.jpg")

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        name1, i1, name2, i2, label = self.pairs[idx]

        img1 = cv2.imread(self._image_path(name1, i1))
        img2 = cv2.imread(self._image_path(name2, i2))

        img1 = self.preprocessing(img1)
        img2 = self.preprocessing(img2)

        return img1, img2, label
