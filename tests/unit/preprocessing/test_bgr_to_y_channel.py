"""Tests for dx_modelzoo.preprocessing.bgr_to_y_channel."""

import numpy as np

from dx_modelzoo.preprocessing.bgr_to_y_channel import BgrToYChannel


class TestBgrToYChannel:
    def test_output_shape(self):
        op = BgrToYChannel()
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        result = op(img)
        assert result.shape == (64, 64, 1)

    def test_output_dtype(self):
        op = BgrToYChannel()
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        assert op(img).dtype == np.float32

    def test_output_range(self):
        op = BgrToYChannel()
        img = (np.random.rand(16, 16, 3) * 255).astype(np.uint8)
        result = op(img)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_accepts_list_input(self):
        op = BgrToYChannel()
        # Must be 3D (H, W, C) for the BGR conversion
        img = [[[0, 0, 0], [255, 255, 255]]]
        result = op(img)
        assert isinstance(result, np.ndarray)
