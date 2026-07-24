"""Tests for dx_modelzoo.preprocessing.mul."""

import numpy as np

from dx_modelzoo.preprocessing.mul import Mul


class TestMul:
    def test_multiplies_by_constant(self):
        op = Mul(x=2.0)
        inp = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(op(inp), [2.0, 4.0, 6.0])

    def test_output_dtype_float32(self):
        op = Mul(x=3.0)
        inp = np.array([1], dtype=np.int32)
        assert op(inp).dtype == np.float32
