"""Tests for dx_modelzoo.preprocessing.add."""

import numpy as np

from dx_modelzoo.preprocessing.add import Add


class TestAdd:
    def test_adds_constant(self):
        op = Add(x=10.0)
        inp = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = op(inp)
        np.testing.assert_array_almost_equal(result, [11.0, 12.0, 13.0])

    def test_output_dtype_float32(self):
        op = Add(x=1.0)
        inp = np.array([0], dtype=np.int32)
        assert op(inp).dtype == np.float32

    def test_negative_constant(self):
        op = Add(x=-5.0)
        inp = np.zeros(3, dtype=np.float32)
        np.testing.assert_array_almost_equal(op(inp), [-5.0, -5.0, -5.0])
