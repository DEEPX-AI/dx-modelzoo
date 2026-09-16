from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.dataset import DATASET_REGISTRY

_INSTALL_GUIDE = """\
  [Market-1501] — Research use only

  1. Download from https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html
     or request access via the project page.
  2. Extract to: <DATA_ROOT>/Market1501/
     Expected structure:
       Market1501/
         bounding_box_test/
           0000_c1s1_000151_01.jpg
           ...
         query/
           0001_c1s1_001051_00.jpg
           ...

  License: Research use only.
  See: https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html
"""


@DATASET_REGISTRY.register
class Market1501(DatasetBase):
    def __init__(self, data_dir: str) -> None:
        self.ensure_exists(data_dir, _INSTALL_GUIDE)
        super().__init__(data_dir)
        self.gallery, self.gallery_ids, self.gallery_cams = self._load_set("bounding_box_test")
        self.query, self.query_ids, self.query_cams = self._load_set("query")
        self._current_set = "gallery"

    def _parse_filename(self, fname: str) -> Tuple[int, int]:
        pid = int(fname.split("_")[0])
        cam = int(fname.split("_")[1][1])
        return pid, cam

    def _load_set(self, subset: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
        set_dir = os.path.join(self.data_dir, subset)
        paths, pids, cams = [], [], []
        for fname in sorted(os.listdir(set_dir)):
            if not fname.endswith(".jpg"):
                continue
            pid, cam = self._parse_filename(fname)
            if pid < 0:
                continue
            paths.append(os.path.join(set_dir, fname))
            pids.append(pid)
            cams.append(cam)
        return paths, np.array(pids), np.array(cams)

    def set_mode(self, mode: str) -> None:
        assert mode in ("gallery", "query")
        self._current_set = mode

    @property
    def _paths(self) -> List[str]:
        return self.gallery if self._current_set == "gallery" else self.query

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int) -> Tuple:
        img = cv2.imread(self._paths[idx])
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {self._paths[idx]}")
        img = self.preprocessing(img)
        return img, idx

    def validate(self) -> None:
        """Validate the dataset's integrity and structure.

        Raises:
            RuntimeError: If the dataset is invalid or corrupted.
        """
        # directory structure
        gallery_dir = os.path.join(self.data_dir, "bounding_box_test")
        query_dir = os.path.join(self.data_dir, "query")
        is_correct_structure = Path(self.data_dir).is_dir() and Path(gallery_dir).is_dir() and Path(query_dir).is_dir()
        if not is_correct_structure:
            raise RuntimeError(
                "Dataset directory structure is invalid. Expected 'bounding_box_test' and 'query' directories under"
                f" {self.data_dir}."
            )

        # file existence
        NUM_GALLERY = 19732
        NUM_QUERY = 3368
        num_gallery = len(list(Path(gallery_dir).glob("*.jpg")))
        num_query = len(list(Path(query_dir).glob("*.jpg")))
        if num_gallery != NUM_GALLERY:
            raise RuntimeError(f"Expected {NUM_GALLERY} images in {gallery_dir}, but found {num_gallery}.")
        if num_query != NUM_QUERY:
            raise RuntimeError(f"Expected {NUM_QUERY} images in {query_dir}, but found {num_query}.")
