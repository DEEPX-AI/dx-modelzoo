from __future__ import annotations

from typing import List, Union

import numpy as np

from dx_modelzoo.postprocessing import POSTPROCESSING_REGISTRY


@POSTPROCESSING_REGISTRY.register("sr_postprocessing")
class SRPostprocessing:
    """Post-processing for ESPCN super resolution models.

    Handles depth-to-space conversion for ONNX output and normalization for DXNN output.
    ONNX output: (batch, upscale_factor^2, h, w) — needs depth-to-space
    DXNN output: (batch, 1, h*upscale_factor, w*upscale_factor) — DepthToSpace already applied
    """

    def __init__(self, upscale_factor: int = 2) -> None:
        self.upscale_factor = upscale_factor

    def __call__(self, x: Union[List[np.ndarray], np.ndarray], **kwargs) -> np.ndarray:
        if isinstance(x, list) and len(x) > 0:
            x = x[0]

        x = x.astype(np.float32)

        if x.ndim == 2:
            # Single 2D output (H, W) — add batch and channel dims
            x = x[np.newaxis, np.newaxis]
        elif x.ndim == 3:
            # (C, H, W) or (B, H, W) — add batch/channel dim
            x = x[np.newaxis]

        batch_size, channels, h, w = x.shape
        expected_channels = self.upscale_factor * self.upscale_factor

        if channels == expected_channels:
            # ONNX format: needs depth-to-space conversion
            x = x.reshape(batch_size, 1, self.upscale_factor, self.upscale_factor, h, w)
            x = x.transpose(0, 1, 4, 2, 5, 3)
            x = x.reshape(batch_size, 1, h * self.upscale_factor, w * self.upscale_factor)
        else:
            # DXNN format — DepthToSpace already applied, normalize if needed.
            # Use threshold > 2.0 to distinguish genuine uint8 [0,255] output
            # from dequantized float [0,~1+ε] where ε is quantization noise.
            if x.max() > 2.0:
                x = x / 255.0

        return np.clip(x, 0.0, 1.0)
