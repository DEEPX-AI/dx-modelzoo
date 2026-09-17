"""Custom postprocessing for EigenPlaces visual place recognition models.

EigenPlaces emits a ``[1, D]`` global descriptor (D = 512 / 2048).
The reference implementation L2-normalizes the descriptor before
matching against the gallery; the postprocessor surfaces that here so
the YAML can declare it explicitly.
"""
from __future__ import annotations

import numpy as np

from dx_modelzoo.postprocessing import POSTPROCESSING_REGISTRY


class L2Normalize:
    """L2-normalize descriptor along the last axis.

    Accepts the raw model output (single tensor or single-element list)
    and returns a unit-norm descriptor of the same shape.
    """

    def __init__(self, axis: int = -1, eps: float = 1e-12) -> None:
        self.axis = axis
        self.eps = float(eps)

    def __call__(self, outputs, **kwargs):
        if isinstance(outputs, list) and len(outputs) == 1:
            outputs = outputs[0]
        if isinstance(outputs, dict):
            outputs = next(iter(outputs.values()))
        x = np.asarray(outputs, dtype=np.float32)
        norm = np.linalg.norm(x, axis=self.axis, keepdims=True)
        return x / np.maximum(norm, self.eps)


if "l2_normalize" not in POSTPROCESSING_REGISTRY:
    POSTPROCESSING_REGISTRY.register("l2_normalize")(L2Normalize)
