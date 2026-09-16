from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.dataset import DATASET_REGISTRY

_INSTALL_GUIDE = """\
  [MVTec Anomaly Detection (MVTec AD)] — Research use only

  1. Register / download from https://www.mvtec.com/company/research/datasets/mvtec-ad
  2. Extract so the directory looks like:
       <eval_path>/
         bottle/
           train/good/000.png ...
           test/good/000.png ...
           test/<defect>/000.png ...
           ground_truth/<defect>/000_mask.png ...
         cable/ ...
         ... (15 categories)

  License: MVTec AD is for non-commercial research/education use only.
"""

_IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp")


def _list_images(directory: str) -> List[str]:
    files: List[str] = []
    for ext in _IMG_EXTS:
        files.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(files)


@DATASET_REGISTRY.register("MVTecAD")
class MVTecADDataset(DatasetBase):
    """MVTec Anomaly Detection dataset.

    Yields one *test* image per index together with its image-level label
    (0 = good/normal, 1 = anomalous), the category name, and the path to the
    pixel-level ground-truth mask (empty string for good images).

    The dataset additionally exposes, per category, the list of ``train/good``
    reference images (:meth:`train_files`) and a helper to preprocess an image
    by path (:meth:`preprocess_path`).  Anomaly-detection evaluators use these
    to build a PatchCore memory bank or EfficientAD normalization statistics
    with the *same* preprocessing / session as the test images.

    Args:
        data_dir: Root of the MVTec AD dataset.
        category: ``None``/``"all"`` -> use every category (metrics are the
            mean over categories).  A single category name, or a list of
            category names, restricts evaluation to that subset.
        max_train_images: Cap on the number of ``train/good`` reference images
            used per category (0 = use all).  Keeps memory-bank / statistics
            building tractable on CPU.
    """

    def __init__(
        self,
        data_dir: str,
        category: Optional[Union[str, List[str]]] = None,
        max_train_images: int = 0,
    ) -> None:
        self.ensure_exists(data_dir, _INSTALL_GUIDE)
        super().__init__(data_dir)

        all_cats = sorted(
            d
            for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d, "test")) and os.path.isdir(os.path.join(data_dir, d, "train"))
        )
        if category is None or category == "" or category == "all":
            cats = all_cats
        elif isinstance(category, (list, tuple)):
            cats = [str(c) for c in category]
        else:
            cats = [str(category)]

        missing = [c for c in cats if c not in all_cats]
        if missing:
            raise ValueError(f"MVTecAD categories not found under {data_dir}: {missing}. Available: {all_cats}")

        self.categories: List[str] = cats
        self.max_train_images = int(max_train_images or 0)

        self.samples: List[Tuple[str, int, str, str]] = []  # (path, label, category, mask_path)
        self._train_files = {}

        for cat in cats:
            cat_dir = os.path.join(data_dir, cat)
            self._train_files[cat] = _list_images(os.path.join(cat_dir, "train", "good"))

            test_dir = os.path.join(cat_dir, "test")
            for defect in sorted(os.listdir(test_dir)):
                ddir = os.path.join(test_dir, defect)
                if not os.path.isdir(ddir):
                    continue
                for ip in _list_images(ddir):
                    if defect == "good":
                        self.samples.append((ip, 0, cat, ""))
                    else:
                        base = os.path.splitext(os.path.basename(ip))[0]
                        mp = os.path.join(cat_dir, "ground_truth", defect, base + "_mask.png")
                        self.samples.append((ip, 1, cat, mp if os.path.exists(mp) else ""))

        if not self.samples:
            raise ValueError(f"No MVTecAD test images found under {data_dir} for categories {cats}")

    # -- helpers used by the anomaly evaluator ---------------------------------
    def _read(self, path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {path}")
        return img

    def preprocess_path(self, path: str) -> np.ndarray:
        """Read + apply the configured preprocessing to the image at ``path``."""
        return self.preprocessing(self._read(path))

    def train_files(self, category: str) -> List[str]:
        files = self._train_files.get(category, [])
        if self.max_train_images > 0:
            return files[: self.max_train_images]
        return files

    # -- Dataset protocol ------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int, str, str]:
        path, label, cat, mask_path = self.samples[idx]
        img = self.preprocess_path(path)
        return img, label, cat, mask_path

    def validate(self) -> None:
        """Validate the dataset's integrity and structure.

        Raises:
            RuntimeError: If the dataset is invalid or corrupted.
        """
        # directory structure
        is_correct_structure = Path(self.data_dir).is_dir() and all(
            Path(cat, "train", "good").is_dir() and Path(cat, "test").is_dir() and Path(cat, "ground_truth").is_dir()
            for cat in Path(self.data_dir).iterdir()
            if cat.is_dir()
        )
        if not is_correct_structure:
            raise RuntimeError(
                "Dataset directory structure is invalid. Expected 'train/good', 'test', and 'ground_truth' directories"
                f" under each category in {self.data_dir}."
            )

        # file existence
        NUM_TRAIN_DATA = 3629
        NUM_TEST_DATA = 1725
        num_train = len(list(Path(self.data_dir).rglob("train/good/*.[jp][pn]g")))
        num_test = len(list(Path(self.data_dir).rglob("test/*/*.[jp][pn]g")))
        if num_train != NUM_TRAIN_DATA:
            raise RuntimeError(f"Expected {NUM_TRAIN_DATA} train images in {self.data_dir}, but found {num_train}.")
        if num_test != NUM_TEST_DATA:
            raise RuntimeError(f"Expected {NUM_TEST_DATA} test images in {self.data_dir}, but found {num_test}.")
