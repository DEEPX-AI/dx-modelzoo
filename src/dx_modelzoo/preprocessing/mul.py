from __future__ import annotations

import numpy as np

from dx_modelzoo.preprocessing import PREPROCESSING_REGISTRY


@PREPROCESSING_REGISTRY.register("mul")
class Mul:
    """Multiply inputs by a constant."""

    def __init__(self, x: float) -> None:
        self.x = x

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        return (inputs * self.x).astype(np.float32)
