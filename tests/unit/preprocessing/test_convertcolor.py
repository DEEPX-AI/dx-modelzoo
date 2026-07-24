"""Tests for dx_modelzoo.preprocessing.convertcolor."""

import numpy as np
import pytest

from dx_modelzoo.preprocessing.convertcolor import ConvertColor


class TestConvertColor:
    def test_bgr2rgb(self):
        op = ConvertColor(form="BGR2RGB")
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[:, :, 0] = 255  # Blue channel
        result = op(img)
        assert result[:, :, 2].mean() == 255  # Now in Red channel

    def test_bgr2gray(self):
        op = ConvertColor(form="BGR2GRAY")
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        result = op(img)
        assert result.shape == (10, 10, 1)

    def test_invalid_form_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            ConvertColor(form="INVALID")

    def test_pil_image_input(self):
        from PIL import Image
        op = ConvertColor(form="RGB2BGR")
        pil_img = Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
        result = op(pil_img)
        assert result.shape == (10, 10, 3)
