from __future__ import annotations

from typing import List

import numpy as np

from dx_modelzoo.preprocessing import PREPROCESSING_REGISTRY


@PREPROCESSING_REGISTRY.register("normalize")
class Normalize:
    """Normalize inputs with mean and std."""

    def __init__(self, mean: List[float], std: List[float]) -> None:
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        mean, std = self.mean, self.std
        # If inputs are channel-first (C,H,W) or (N,C,H,W), reshape for broadcasting
        if inputs.ndim >= 3 and inputs.shape[-1] != len(mean):
            shape = [1] * inputs.ndim
            shape[-3] = len(mean)
            mean = mean.reshape(shape)
            std = std.reshape(shape)
        return (inputs - mean) / std
