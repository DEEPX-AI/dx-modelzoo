"""Tests for dx_modelzoo.postprocessing.segmentation_argmax."""

import numpy as np

from dx_modelzoo.postprocessing.segmentation_argmax import SegmentationArgmax


class TestSegmentationArgmax:
    def test_nchw_basic(self):
        op = SegmentationArgmax(layout="nchw")
        # 2 classes, 4x4
        logits = np.zeros((1, 2, 4, 4), dtype=np.float32)
        logits[0, 1, :, :] = 1.0  # class 1 everywhere
        result = op([logits])
        assert result.shape == (1, 4, 4)
        assert (result == 1).all()

    def test_nhwc_basic(self):
        op = SegmentationArgmax(layout="nhwc")
        logits = np.zeros((1, 4, 4, 2), dtype=np.float32)
        logits[0, :, :, 0] = 1.0  # class 0 everywhere
        result = op([logits])
        assert result.shape == (1, 4, 4)
        assert (result == 0).all()

    def test_target_size_resize(self):
        op = SegmentationArgmax(layout="nchw", target_size=[8, 8])
        logits = np.zeros((1, 3, 4, 4), dtype=np.float32)
        logits[0, 2, :, :] = 1.0
        result = op([logits])
        assert result.shape == (1, 8, 8)
