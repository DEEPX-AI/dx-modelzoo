"""Tests for dx_modelzoo.preprocessing.transpose."""

import numpy as np

from dx_modelzoo.preprocessing.transpose import Transpose


class TestTranspose:
    def test_hwc_to_chw(self):
        op = Transpose(axis=[2, 0, 1])
        inp = np.zeros((32, 32, 3), dtype=np.float32)
        result = op(inp)
        assert result.shape == (3, 32, 32)

    def test_contiguous(self):
        op = Transpose(axis=[2, 0, 1])
        inp = np.zeros((10, 10, 3), dtype=np.float32)
        result = op(inp)
        assert result.flags["C_CONTIGUOUS"]

    def test_identity_transpose(self):
        op = Transpose(axis=[0, 1, 2])
        inp = np.arange(24).reshape(2, 3, 4)
        result = op(inp)
        np.testing.assert_array_equal(result, inp)
