"""Tests for dx_modelzoo.postprocessing.coord_scaler."""

import numpy as np

from dx_modelzoo.postprocessing.coord_scaler import unpad_and_scale


class TestUnpadAndScale:
    def test_direct_resize_scaling(self):
        # Model 640x640, original 480x640 (no pad)
        boxes = np.array([[0, 0, 640, 640]], dtype=np.float32)
        result = unpad_and_scale(boxes, model_hw=(640, 640), orig_hw=(480, 640), pad_resize=False)
        # Should scale to original image
        assert result[0, 2] <= 640  # x2 <= orig_w
        assert result[0, 3] <= 480  # y2 <= orig_h

    def test_empty_boxes(self):
        boxes = np.zeros((0, 4), dtype=np.float32)
        result = unpad_and_scale(boxes, model_hw=(640, 640), orig_hw=(480, 640))
        assert result.shape == (0, 4)

    def test_pad_resize_scaling(self):
        # Letterbox: 640x640 with 480x640 original → padded on top/bottom
        boxes = np.array([[100, 100, 200, 200]], dtype=np.float32)
        result = unpad_and_scale(boxes, model_hw=(640, 640), orig_hw=(480, 640), pad_resize=True)
        assert result.shape == (1, 4)
