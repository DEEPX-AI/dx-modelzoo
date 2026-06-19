"""Tests for dx_modelzoo.preprocessing.normalize."""

import numpy as np

from dx_modelzoo.preprocessing.normalize import Normalize


class TestNormalize:
    def test_basic_normalize(self):
        op = Normalize(mean=[0.5], std=[0.5])
        inp = np.array([1.0], dtype=np.float32)
        result = op(inp)
        np.testing.assert_almost_equal(result, [1.0])  # (1-0.5)/0.5 = 1.0

    def test_imagenet_shape_hwc(self):
        op = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        inp = np.ones((224, 224, 3), dtype=np.float32)
        result = op(inp)
        assert result.shape == (224, 224, 3)

    def test_channel_first_broadcast(self):
        op = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        inp = np.ones((3, 32, 32), dtype=np.float32)
        result = op(inp)
        # (1 - 0.5) / 0.5 = 1.0
        np.testing.assert_array_almost_equal(result, np.ones((3, 32, 32)))

    def test_4d_input(self):
        op = Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
        inp = np.ones((2, 3, 16, 16), dtype=np.float32)
        result = op(inp)
        assert result.shape == (2, 3, 16, 16)
