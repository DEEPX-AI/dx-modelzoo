"""Tests for dx_modelzoo.dataset.bsd."""

import os
import numpy as np
import cv2
import pytest

from dx_modelzoo.dataset.bsd import BSD68, BSD100, CBSD68
from dx_modelzoo.preprocessing import PreprocessingPipeline


class TestBSD68:
    def _setup_data(self, tmp_path):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / "test001.png"), img)
        return tmp_path

    def test_init_and_len(self, tmp_path):
        data_dir = self._setup_data(tmp_path)
        ds = BSD68(str(data_dir))
        assert len(ds) >= 1

    def test_is_subclass(self):
        from dx_modelzoo.common.dataloader import DatasetBase
        assert issubclass(BSD68, DatasetBase)
        assert issubclass(BSD100, DatasetBase)
        assert issubclass(CBSD68, DatasetBase)
