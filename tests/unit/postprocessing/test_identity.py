"""Tests for dx_modelzoo.postprocessing.identity."""

import numpy as np

from dx_modelzoo.postprocessing.identity import Identity


class TestIdentity:
    def test_unwraps_single_element_list(self):
        op = Identity()
        data = [np.array([1, 2, 3])]
        result = op(data)
        np.testing.assert_array_equal(result, np.array([1, 2, 3]))

    def test_multi_element_list_unchanged(self):
        op = Identity()
        data = [np.array([1]), np.array([2])]
        result = op(data)
        assert len(result) == 2

    def test_non_list_passthrough(self):
        op = Identity()
        data = np.array([1, 2, 3])
        result = op(data)
        np.testing.assert_array_equal(result, data)
