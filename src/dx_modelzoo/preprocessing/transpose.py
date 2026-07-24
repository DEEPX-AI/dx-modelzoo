from __future__ import annotations

from typing import List

import numpy as np

from dx_modelzoo.preprocessing import PREPROCESSING_REGISTRY


@PREPROCESSING_REGISTRY.register("transpose")
class Transpose:
    """Transpose axes of the input array."""

    def __init__(self, axis: List[int]) -> None:
        self.axis = axis

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(np.transpose(inputs, axes=self.axis))
