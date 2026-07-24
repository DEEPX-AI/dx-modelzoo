"""Tests for dx_modelzoo.dataset.ade20k."""

import os
import numpy as np
import cv2
import pytest

from dx_modelzoo.dataset.ade20k import ADE20K
from dx_modelzoo.preprocessing import PreprocessingPipeline


class TestADE20K:
    def _setup_data(self, tmp_path):
        img_dir = tmp_path / "images" / "validation"
        ann_dir = tmp_path / "annotations" / "validation"
        img_dir.mkdir(parents=True)
        ann_dir.mkdir(parents=True)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / "ADE_val_00000001.jpg"), img)
        label = np.ones((100, 100), dtype=np.uint8)  # class 1
        cv2.imwrite(str(ann_dir / "ADE_val_00000001.png"), label)
        return tmp_path

    def test_init_and_len(self, tmp_path):
        data_dir = self._setup_data(tmp_path)
        ds = ADE20K(str(data_dir))
        assert len(ds) == 1

    def test_getitem(self, tmp_path):
        data_dir = self._setup_data(tmp_path)
        ds = ADE20K(str(data_dir), eval_size=64)
        ds.preprocessing = PreprocessingPipeline([{"type": "resize", "size": [64, 64]}, {"type": "div", "x": 255}])
        img, label = ds[0]
        assert isinstance(img, np.ndarray)
        assert label.shape == (64, 64)
