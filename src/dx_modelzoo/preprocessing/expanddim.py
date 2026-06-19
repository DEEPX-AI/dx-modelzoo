from __future__ import annotations

import numpy as np

from dx_modelzoo.preprocessing import PREPROCESSING_REGISTRY


@PREPROCESSING_REGISTRY.register("expanddim")
class ExpandDim:
    """Expand dimensions by inserting a new axis."""

    def __init__(self, axis: int) -> None:
        self.axis = axis

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        return np.expand_dims(inputs, axis=self.axis)
