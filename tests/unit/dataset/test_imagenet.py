"""Tests for dx_modelzoo.dataset.imagenet."""

import os
import numpy as np
import cv2
import pytest

from dx_modelzoo.dataset.imagenet import ILSVRC2012
from dx_modelzoo.preprocessing import PreprocessingPipeline


class TestILSVRC2012:
    def _setup_data(self, tmp_path):
        # Create minimal structure: val/n01440764/img.JPEG
        cls_dir = tmp_path / "n01440764"
        cls_dir.mkdir()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(cls_dir / "test.JPEG"), img)
        cls_dir2 = tmp_path / "n01443537"
        cls_dir2.mkdir()
        cv2.imwrite(str(cls_dir2 / "test2.JPEG"), img)
        return tmp_path

    def test_init_and_len(self, tmp_path):
        data_dir = self._setup_data(tmp_path)
        ds = ILSVRC2012(str(data_dir))
        assert len(ds) == 2

    def test_getitem(self, tmp_path):
        data_dir = self._setup_data(tmp_path)
        ds = ILSVRC2012(str(data_dir))
        ds.preprocessing = PreprocessingPipeline([{"type": "resize", "size": [224, 224]}, {"type": "div", "x": 255}])
        img, label = ds[0]
        assert isinstance(img, np.ndarray)
        assert isinstance(label, int)

    def test_class_map(self, tmp_path):
        data_dir = self._setup_data(tmp_path)
        ds = ILSVRC2012(str(data_dir))
        assert len(ds.class_map) == 2

    def test_missing_dir_raises(self):
        with pytest.raises(FileNotFoundError):
            ILSVRC2012("/nonexistent/imagenet")
