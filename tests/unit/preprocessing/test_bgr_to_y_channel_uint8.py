"""Tests for dx_modelzoo.preprocessing.bgr_to_y_channel_uint8."""

import numpy as np

from dx_modelzoo.preprocessing.bgr_to_y_channel_uint8 import BgrToYChannelUint8


class TestBgrToYChannelUint8:
    def test_output_shape(self):
        op = BgrToYChannelUint8()
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        result = op(img)
        assert result.shape == (64, 64, 1)

    def test_output_dtype(self):
        op = BgrToYChannelUint8()
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        assert op(img).dtype == np.uint8

    def test_output_range(self):
        op = BgrToYChannelUint8()
        img = (np.random.rand(16, 16, 3) * 255).astype(np.uint8)
        result = op(img)
        assert result.min() >= 0
        assert result.max() <= 255
