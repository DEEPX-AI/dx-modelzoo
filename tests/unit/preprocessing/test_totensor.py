"""Tests for dx_modelzoo.preprocessing.totensor."""

import numpy as np
import pytest

from dx_modelzoo.preprocessing.totensor import ToTensor


class TestToTensor:
    def test_hwc_to_chw(self):
        op = ToTensor()
        inp = np.zeros((32, 32, 3), dtype=np.uint8)
        result = op(inp)
        assert result.shape == (3, 32, 32)
        assert result.dtype == np.float32

    def test_2d_to_1hw(self):
        op = ToTensor()
        inp = np.zeros((64, 64), dtype=np.uint8)
        result = op(inp)
        assert result.shape == (1, 64, 64)

    def test_4d_nhwc_to_nchw(self):
        op = ToTensor()
        inp = np.zeros((2, 32, 32, 3), dtype=np.uint8)
        result = op(inp)
        assert result.shape == (2, 3, 32, 32)

    def test_scales_to_0_1(self):
        op = ToTensor()
        inp = np.full((10, 10, 3), 255, dtype=np.uint8)
        result = op(inp)
        np.testing.assert_almost_equal(result.max(), 1.0)

    def test_invalid_ndim_raises(self):
        op = ToTensor()
        inp = np.zeros((2, 3, 4, 5, 6), dtype=np.uint8)
        with pytest.raises(ValueError, match="only supports"):
            op(inp)
