"""Tests for dx_modelzoo.preprocessing.resize."""

import numpy as np

from dx_modelzoo.preprocessing.resize import Resize


class TestResize:
    def test_default_mode_resize(self):
        op = Resize(size=[100, 100])
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        result = op(img)
        assert result.shape[0] == 100
        assert result.shape[1] == 100

    def test_torchvision_mode(self):
        op = Resize(mode="torchvision", size=[128, 128], interpolation="BILINEAR")
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        result = op(img)
        # torchvision mode resizes short side, so output depends on aspect
        assert result.shape[0] >= 128 or result.shape[1] >= 128

    def test_preserves_channels(self):
        op = Resize(size=[64, 64])
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = op(img)
        assert result.shape[2] == 3

    def test_pil_image_input(self):
        from PIL import Image
        op = Resize(size=[50, 50])
        pil_img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        result = op(pil_img)
        assert isinstance(result, np.ndarray)
