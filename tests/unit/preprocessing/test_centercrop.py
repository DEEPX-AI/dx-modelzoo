"""Tests for dx_modelzoo.preprocessing.centercrop."""

import numpy as np

from dx_modelzoo.preprocessing.centercrop import CenterCrop


class TestCenterCrop:
    def test_basic_crop(self):
        op = CenterCrop(height=100, width=100)
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        result = op(img)
        assert result.shape == (100, 100, 3)

    def test_crop_is_centered(self):
        op = CenterCrop(height=2, width=2)
        img = np.arange(16).reshape(4, 4).astype(np.float32)
        result = op(img)
        # Center of 4x4: rows 1-2, cols 1-2
        np.testing.assert_array_equal(result, [[5, 6], [9, 10]])

    def test_pil_image_input(self):
        from PIL import Image
        op = CenterCrop(height=10, width=10)
        pil_img = Image.fromarray(np.zeros((20, 20, 3), dtype=np.uint8))
        result = op(pil_img)
        assert result.shape == (10, 10, 3)
