from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from dx_modelzoo.preprocessing import PREPROCESSING_REGISTRY

COLOR_CONVERT_MAP = {
    "BGR2RGB": cv2.COLOR_BGR2RGB,
    "RGB2BGR": cv2.COLOR_RGB2BGR,
    "BGR2GRAY": cv2.COLOR_BGR2GRAY,
    "RGB2GRAY": cv2.COLOR_RGB2GRAY,
    "GRAY2BGR": cv2.COLOR_GRAY2BGR,
    "GRAY2RGB": cv2.COLOR_GRAY2RGB,
}


@PREPROCESSING_REGISTRY.register("convertcolor")
class ConvertColor:
    """Convert color space."""

    def __init__(self, form: str) -> None:
        if form not in COLOR_CONVERT_MAP:
            raise ValueError(f"Unsupported color conversion: {form}. Available: {list(COLOR_CONVERT_MAP.keys())}")
        self.form = form
        self.code = COLOR_CONVERT_MAP[form]

    def __call__(self, inputs) -> np.ndarray:
        if isinstance(inputs, Image.Image):
            inputs = np.array(inputs)
        result = cv2.cvtColor(inputs, self.code)
        if result.ndim == 2:
            result = result[:, :, np.newaxis]
        return result
