from __future__ import annotations

import numpy as np

from dx_modelzoo.preprocessing import PREPROCESSING_REGISTRY


@PREPROCESSING_REGISTRY.register("totensor")
class ToTensor:
    """Mimics ``torchvision.transforms.ToTensor()``.

    Converts a HWC uint8 [0, 255] numpy array to CHW float32 [0, 1].
    Equivalent to ``div(x=255)`` + ``transpose``, but as a single step.
    """

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        if inputs.ndim == 2:
            # (H, W) → (1, H, W)
            inputs = inputs[np.newaxis, :, :]
        elif inputs.ndim == 3:
            # (H, W, C) → (C, H, W)
            inputs = np.transpose(inputs, (2, 0, 1))
        elif inputs.ndim == 4:
            # (N, H, W, C) → (N, C, H, W)
            inputs = np.transpose(inputs, (0, 3, 1, 2))
        else:
            raise ValueError(f"ToTensor only supports 2D, 3D, 4D inputs, got {inputs.ndim}D")
        return (inputs / 255.0).astype(np.float32)
