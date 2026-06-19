from __future__ import annotations

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.dataset import DATASET_REGISTRY

_INSTALL_GUIDE = """\
  [HPatches] — Research use

  1. Download from https://github.com/hpatches/hpatches-dataset
     - hpatches-sequences-release.tar.gz
  2. Extract to: <DATA_ROOT>/hpatches/
     Expected structure:
       hpatches/
         hpatches-sequences-release/
           i_ajuntament/
             1.ppm 2.ppm 3.ppm 4.ppm 5.ppm 6.ppm
             H_1_2 H_1_3 H_1_4 H_1_5 H_1_6
           v_*/
             ...

  Sequences prefixed with 'i_' = illumination changes (same viewpoint).
  Sequences prefixed with 'v_' = viewpoint changes.

  License: CC BY-SA (research use).
  See: https://github.com/hpatches/hpatches-dataset
"""


def _load_homography(seq_dir: str, n: int) -> Optional[np.ndarray]:
    """Load ground-truth homography ``H_1_N`` for image *n*.

    Returns ``None`` for the reference image (n=1).
    """
    if n == 1:
        return None
    path = os.path.join(seq_dir, f"H_1_{n}")
    if not os.path.isfile(path):
        return None
    return np.loadtxt(path).astype(np.float64)


@DATASET_REGISTRY.register
class HPatches(DatasetBase):
    """HPatches sequences dataset.

    Each sample is one of the 6 ``.ppm`` images in a sequence.  The
    homography ``H_1_N`` mapping the reference image (1.ppm) to the
    N-th image is loaded and exposed via the returned tuple, so
    feature-extraction / VPR evaluators can compute matching metrics
    against the reference image.

    Return tuple: ``(preprocessed_img, orig_shape, seq, n, path, H_1_n)``
    where ``H_1_n`` is a 3×3 ``ndarray`` or ``None`` for the reference
    image (n=1).
    """

    def __init__(self, data_dir: str) -> None:
        self.ensure_exists(data_dir, _INSTALL_GUIDE)
        super().__init__(data_dir)
        seqs = sorted(d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d)))
        self.samples: List[Tuple[str, str, int, str]] = []
        for seq in seqs:
            seq_dir = os.path.join(self.data_dir, seq)
            for n in range(1, 7):
                img = os.path.join(seq_dir, f"{n}.ppm")
                if os.path.isfile(img):
                    self.samples.append((seq, img, n, seq_dir))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple:
        seq, img_path, n, seq_dir = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {img_path}")
        H_1_n = _load_homography(seq_dir, n)
        return self.preprocessing(img), img.shape, seq, n, img_path, H_1_n
