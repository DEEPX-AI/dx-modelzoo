"""ViTPose heatmap decoder.

Decodes Gaussian heatmaps ``[1, K, Hh, Wh]`` to keypoint coordinates in
model-input pixel space (width ``model_w``, height ``model_h``), following the
standard mmpose ``keypoints_from_heatmaps`` argmax + 0.25 sub-pixel refinement.
"""
from __future__ import annotations

import numpy as np

from dx_modelzoo.postprocessing import POSTPROCESSING_REGISTRY


@POSTPROCESSING_REGISTRY.register("vitpose_heatmap_decode")
class ViTPoseHeatmapDecode:
    """Decode top-down heatmaps to keypoints.

    Input: heatmaps ``[1, K, Hh, Wh]`` (passed as the single model output, a
    list/tuple of one tensor or a bare ndarray).
    Output: ``(keypoints [K, 2], scores [K])`` in model-input pixel coords.
    """

    def __init__(self, model_w: float = 192.0, model_h: float = 256.0, **kwargs) -> None:
        self.model_w = float(model_w)
        self.model_h = float(model_h)

    def __call__(self, outputs, **kwargs):
        heatmaps = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        heatmaps = np.asarray(heatmaps)
        if heatmaps.ndim == 4:
            heatmaps = heatmaps[0]
        k, h, w = heatmaps.shape

        flat = heatmaps.reshape(k, -1)
        idx = flat.argmax(axis=1)
        scores = flat.max(axis=1).astype(np.float32)
        x_idx = (idx % w).astype(np.float32)
        y_idx = (idx // w).astype(np.float32)

        # 0.25 sub-pixel refinement toward the higher-valued neighbour.
        for i in range(k):
            px, py = int(x_idx[i]), int(y_idx[i])
            if 1 < px < w - 1:
                dx = heatmaps[i, py, px + 1] - heatmaps[i, py, px - 1]
                x_idx[i] += 0.25 * np.sign(dx)
            if 1 < py < h - 1:
                dy = heatmaps[i, py + 1, px] - heatmaps[i, py - 1, px]
                y_idx[i] += 0.25 * np.sign(dy)

        kpts = np.empty((k, 2), dtype=np.float32)
        kpts[:, 0] = x_idx * (self.model_w / w)
        kpts[:, 1] = y_idx * (self.model_h / h)
        return kpts, scores
