from __future__ import annotations

import os
from typing import Callable, List, Tuple

import numpy as np

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.dataset import DATASET_REGISTRY

_INSTALL_GUIDE = """\
  [NYU Depth v2] — Research use only

  1. Download from https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
     or preprocessed version from:
     https://github.com/fangchangma/sparse-to-dense/
  2. The preprocessed HDF5 version is required:
     - Each .h5 file contains 'rgb' and 'depth' arrays.
  3. Extract to: <DATA_ROOT>/nyudepthv2/
     Expected structure:
       nyudepthv2/
         val/
           official/
             00001.h5
             00002.h5
             ...
     NOTE: the loader treats each sub-directory of its ``eval_path`` as a
     class and walks it for ``.h5`` files, so point ``eval_path`` at
     ``nyudepthv2/val`` (which contains ``official/``).

  Requires: h5py (for loading .h5 files)
  License: Research use only.
  See: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
"""


@DATASET_REGISTRY.register
class NYUDepthv2(DatasetBase):
    def __init__(self, data_dir: str) -> None:
        self.ensure_exists(data_dir, _INSTALL_GUIDE)
        super().__init__(data_dir)
        classes, class_to_idx = self._find_classes()
        self.imgs = self._make_dataset(class_to_idx)
        self.classes = classes
        self.class_to_idx = class_to_idx
        self._label_preprocessing = None

    @property
    def label_preprocessing(self) -> Callable:
        if self._label_preprocessing is None:
            raise ValueError("Label preprocessing is not set.")
        return self._label_preprocessing

    @label_preprocessing.setter
    def label_preprocessing(self, value: Callable) -> None:
        self._label_preprocessing = value

    def _h5_loader(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required for NYU dataset")
        with h5py.File(path, "r") as h5f:
            rgb = np.array(h5f["rgb"])
            rgb = np.transpose(rgb, (1, 2, 0))
            depth = np.array(h5f["depth"])
        return rgb, depth

    def _find_classes(self) -> Tuple[List[str], dict]:
        classes = sorted(d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d)))
        return classes, {c: i for i, c in enumerate(classes)}

    def _make_dataset(self, class_to_idx: dict) -> List[Tuple[str, int]]:
        images = []
        for target in sorted(os.listdir(self.data_dir)):
            d = os.path.join(self.data_dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if fname.endswith(".h5"):
                        images.append((os.path.join(root, fname), class_to_idx[target]))
        return images

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, index: int) -> Tuple:
        path, _ = self.imgs[index]
        rgb, depth = self._h5_loader(path)
        label = self._label_preprocessing(depth) if self._label_preprocessing is not None else depth
        return self.preprocessing(rgb), label
