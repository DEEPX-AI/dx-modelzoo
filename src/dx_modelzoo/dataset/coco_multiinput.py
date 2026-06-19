from __future__ import annotations

import glob
import os
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.dataset import DATASET_REGISTRY

_INSTALL_GUIDE = """\
  [COCOMultiInput] expects COCO val2017 layout under ``data_dir``:
    coco/
      val2017/*.jpg
"""


@DATASET_REGISTRY.register
class COCOMultiInput(DatasetBase):
    """COCO val2017 wrapper that produces a multi-input dict.

    Loads each image from ``val2017/*.jpg`` and feeds it to the
    pipeline as the named tensor ``image_input_name``.  Any additional
    inputs declared in the YAML ``inputs`` spec are filled with the
    constant ``value`` field (or zero if absent), making it usable for
    detectors with embedded NMS that take control scalars
    (e.g. BlazeFace: conf_threshold / max_detections / iou_threshold).
    """

    def __init__(
        self,
        data_dir: str,
        inputs: List[Dict] = None,
        image_input_name: str = "image",
        num_samples: int = 0,
    ) -> None:
        self.ensure_exists(data_dir, _INSTALL_GUIDE)
        super().__init__(data_dir)
        self.inputs_spec: List[Dict] = inputs or []
        self.image_input_name = image_input_name
        files = sorted(glob.glob(os.path.join(data_dir, "val2017", "*.jpg")))
        if num_samples and len(files) > num_samples:
            files = files[:num_samples]
        self.img_files = files

    def __len__(self) -> int:
        return len(self.img_files)

    def _make_scalar(self, spec: Dict) -> np.ndarray:
        from dx_modelzoo.dataset.synthetic import _DTYPE_MAP

        shape = tuple(spec.get("shape", [1]))
        dtype = _DTYPE_MAP.get(str(spec.get("dtype", "float32")), np.float32)
        value = spec.get("value", 0)
        return np.full(shape, value, dtype=dtype)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, Any], int]:
        img = cv2.imread(self.img_files[idx])
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {self.img_files[idx]}")
        img = self.preprocessing(img)
        sample: Dict[str, np.ndarray] = {self.image_input_name: img}
        for spec in self.inputs_spec:
            name = str(spec["name"])
            if name == self.image_input_name:
                continue
            sample[name] = self._make_scalar(spec)
        return sample, idx
