from __future__ import annotations

import numpy as np
from PIL import Image

from dx_modelzoo.preprocessing import PREPROCESSING_REGISTRY


@PREPROCESSING_REGISTRY.register("centercrop")
class CenterCrop:
    """Crop the center of the image."""

    def __init__(self, height: int, width: int) -> None:
        self.height = height
        self.width = width

    def __call__(self, inputs) -> np.ndarray:
        if isinstance(inputs, Image.Image):
            inputs = np.array(inputs)
        h, w = inputs.shape[:2]
        left = int(round((w - self.width) / 2.0))
        top = int(round((h - self.height) / 2.0))
        return inputs[top : top + self.height, left : left + self.width]
