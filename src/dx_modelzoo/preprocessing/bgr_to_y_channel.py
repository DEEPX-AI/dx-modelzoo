from __future__ import annotations

import numpy as np

from dx_modelzoo.preprocessing import PREPROCESSING_REGISTRY


@PREPROCESSING_REGISTRY.register("bgr_to_y_channel")
class BgrToYChannel:
    """Convert BGR image to Y channel (luminance) in YCbCr color space. Returns [0, 1] float32."""

    def __call__(self, image) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        image = image.astype(np.float32) / 255.0
        ycbcr = np.matmul(
            image,
            [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]],
        ) + [16, 128, 128]
        ycbcr /= 255.0
        # 0:1 slice keeps 3D shape (H, W, 1) so transpose [2,0,1] works correctly
        return ycbcr[:, :, 0:1].astype(np.float32)
