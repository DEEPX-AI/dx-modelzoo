import os
from typing import List, Tuple

import cv2
import numpy as np

from dx_modelzoo.dataset import DatasetBase


class Market1501Dataset(DatasetBase):
    """Market-1501 dataset for person re-identification.

    Provides gallery and query sets. The evaluator extracts embeddings
    from both sets and computes CMC / mAP metrics.

    Dataset structure:
        dataset_root/
            bounding_box_test/
                PPPP_cCsCd_FFFFFF_NN.jpg
            query/
                PPPP_cCsCd_FFFFFF_NN.jpg

    Args:
        data_dir (str): dataset root dir.
    """

    def __init__(self, data_dir: str):
        super().__init__(data_dir)
        self.gallery, self.gallery_ids, self.gallery_cams = self._load_set("bounding_box_test")
        self.query, self.query_ids, self.query_cams = self._load_set("query")
        self._current_set = "gallery"

    def _parse_filename(self, fname: str) -> Tuple[int, int]:
        """Extract person_id and camera_id from filename."""
        pid = int(fname.split("_")[0])
        cam = int(fname.split("_")[1][1])  # cXsYd -> X
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

    def set_mode(self, mode: str):
        """Switch between 'gallery' and 'query' iteration."""
        assert mode in ("gallery", "query")
        self._current_set = mode

    @property
    def _paths(self):
        return self.gallery if self._current_set == "gallery" else self.query

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx) -> Tuple:
        img = cv2.imread(self._paths[idx])
        img = self.preprocessing(img)
        return img, idx
