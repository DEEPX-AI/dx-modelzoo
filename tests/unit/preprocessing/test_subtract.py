"""Tests for dx_modelzoo.preprocessing.subtract."""

import numpy as np

from dx_modelzoo.preprocessing.subtract import Subtract


class TestSubtract:
    def test_subtracts_constant(self):
        op = Subtract(x=10.0)
        inp = np.array([15.0, 20.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(op(inp), [5.0, 10.0])

    def test_output_dtype_float32(self):
        op = Subtract(x=1.0)
        inp = np.array([5], dtype=np.int32)
        assert op(inp).dtype == np.float32
