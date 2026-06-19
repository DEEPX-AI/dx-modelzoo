"""Tests for dx_modelzoo.preprocessing.expanddim."""

import numpy as np

from dx_modelzoo.preprocessing.expanddim import ExpandDim


class TestExpandDim:
    def test_expand_axis_0(self):
        op = ExpandDim(axis=0)
        inp = np.zeros((3, 224, 224), dtype=np.float32)
        result = op(inp)
        assert result.shape == (1, 3, 224, 224)

    def test_expand_axis_negative(self):
        op = ExpandDim(axis=-1)
        inp = np.zeros((3, 4), dtype=np.float32)
        result = op(inp)
        assert result.shape == (3, 4, 1)
