"""Tests for dx_modelzoo.postprocessing.masks."""

import numpy as np

from dx_modelzoo.postprocessing.masks import crop_mask


class TestCropMask:
    def test_basic_crop(self):
        masks = np.ones((1, 100, 100), dtype=np.float32)
        boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
        result = crop_mask(masks, boxes)
        # Outside the box should be zero
        assert result[0, 0, 0] == 0.0
        # Inside should be preserved
        assert result[0, 20, 20] == 1.0

    def test_empty_masks(self):
        masks = np.zeros((0, 100, 100), dtype=np.float32)
        boxes = np.zeros((0, 4), dtype=np.float32)
        result = crop_mask(masks, boxes)
        assert result.shape == (0, 100, 100)

    def test_multiple_masks(self):
        masks = np.ones((3, 50, 50), dtype=np.float32)
        boxes = np.array([
            [0, 0, 25, 25],
            [10, 10, 40, 40],
            [0, 0, 50, 50],
        ], dtype=np.float32)
        result = crop_mask(masks, boxes)
        assert result.shape == (3, 50, 50)
