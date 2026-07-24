from __future__ import annotations

import numpy as np

from dx_modelzoo.preprocessing import PREPROCESSING_REGISTRY


@PREPROCESSING_REGISTRY.register("bgr_to_y_channel_uint8")
class BgrToYChannelUint8:
    """Convert BGR image to Y channel. Returns [0, 255] uint8 with shape (H, W, 1)."""

    def __call__(self, image) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        image = image.astype(np.float32) / 255.0
        ycbcr = np.matmul(
            image,
            [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]],
        ) + [16, 128, 128]
        # 0:1 slice keeps 3D shape (H, W, 1) so a later transpose [2,0,1] works correctly
        return np.clip(ycbcr[:, :, 0:1], 0, 255).astype(np.uint8)
