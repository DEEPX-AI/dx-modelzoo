"""Tests for dx_modelzoo.dataset.cityscapes."""

import os
import numpy as np
import cv2
import pytest

from dx_modelzoo.dataset.cityscapes import Cityscapes
from dx_modelzoo.preprocessing import PreprocessingPipeline


class TestCityscapes:
    def _setup_data(self, tmp_path):
        # Cityscapes expects val.txt with img,label pairs
        img_dir = tmp_path / "leftImg8bit" / "val" / "city1"
        lbl_dir = tmp_path / "gtFine" / "val" / "city1"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / "img001.png"), img)
        label = np.zeros((100, 200), dtype=np.uint8)
        cv2.imwrite(str(lbl_dir / "lbl001.png"), label)
        # Write val.txt
        val_txt = tmp_path / "val.txt"
        val_txt.write_text("leftImg8bit/val/city1/img001.png,gtFine/val/city1/lbl001.png\n")
        return tmp_path

    def test_init_and_len(self, tmp_path):
        data_dir = self._setup_data(tmp_path)
        ds = Cityscapes(str(data_dir))
        assert len(ds) == 1

    def test_getitem(self, tmp_path):
        data_dir = self._setup_data(tmp_path)
        ds = Cityscapes(str(data_dir))
        ds.preprocessing = PreprocessingPipeline([{"type": "resize", "size": [64, 64]}, {"type": "div", "x": 255}])
        img, label = ds[0]
        assert isinstance(img, np.ndarray)
        assert isinstance(label, np.ndarray)

    def test_is_subclass(self):
        from dx_modelzoo.common.dataloader import DatasetBase
        assert issubclass(Cityscapes, DatasetBase)
