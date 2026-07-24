from __future__ import annotations

from typing import List, Optional

import numpy as np

from dx_modelzoo.postprocessing import POSTPROCESSING_REGISTRY


@POSTPROCESSING_REGISTRY.register("topk")
class TopK:
    """Return top-k indices from model output."""

    def __init__(self, k: Optional[List[int]] = None, skip_background: bool = False) -> None:
        self.k = k or [1, 5]
        self.skip_background = skip_background

    def __call__(self, outputs, **kwargs):
        output = outputs[0] if isinstance(outputs, list) else outputs
        output = np.squeeze(output)
        if self.skip_background:
            output = output[..., 1:]
        sorted_indices = np.argsort(output, axis=-1)[..., ::-1]
        return sorted_indices[..., : max(self.k)]
