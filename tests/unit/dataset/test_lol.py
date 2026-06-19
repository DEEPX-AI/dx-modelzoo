"""Tests for dx_modelzoo.dataset.lol."""

import os
import numpy as np
import cv2
import pytest

from dx_modelzoo.dataset.lol import LOL
from dx_modelzoo.preprocessing import PreprocessingPipeline


class TestLOL:
    def _setup_data(self, tmp_path):
        # LOL expects eval15/low/*.png and eval15/high/*.png
        low_dir = tmp_path / "eval15" / "low"
        high_dir = tmp_path / "eval15" / "high"
        low_dir.mkdir(parents=True)
        high_dir.mkdir(parents=True)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(low_dir / "1.png"), img)
        cv2.imwrite(str(high_dir / "1.png"), img)
        return tmp_path

    def test_init_and_len(self, tmp_path):
        data_dir = self._setup_data(tmp_path)
        ds = LOL(str(data_dir))
        assert len(ds) == 1

    def test_getitem(self, tmp_path):
        data_dir = self._setup_data(tmp_path)
        ds = LOL(str(data_dir))
        ds.preprocessing = PreprocessingPipeline([{"type": "resize", "size": [64, 64]}, {"type": "div", "x": 255}])
        low_img, high_img = ds[0]
        assert isinstance(low_img, np.ndarray)

    def test_is_subclass(self):
        from dx_modelzoo.common.dataloader import DatasetBase
        assert issubclass(LOL, DatasetBase)
