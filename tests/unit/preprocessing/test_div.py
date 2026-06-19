"""Tests for dx_modelzoo.preprocessing.div."""

import numpy as np

from dx_modelzoo.preprocessing.div import Div


class TestDiv:
    def test_divides_by_constant(self):
        op = Div(x=2.0)
        inp = np.array([4.0, 6.0, 8.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(op(inp), [2.0, 3.0, 4.0])

    def test_div_255_normalizes(self):
        op = Div(x=255.0)
        inp = np.array([0, 128, 255], dtype=np.uint8)
        result = op(inp)
        assert result.dtype == np.float32
        assert result[0] == 0.0
        np.testing.assert_almost_equal(result[2], 1.0)
