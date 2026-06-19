"""RTMPose SimCC decoder."""
from __future__ import annotations

import numpy as np

from dx_modelzoo.postprocessing import POSTPROCESSING_REGISTRY

_SIMCC_SPLIT_RATIO = 2.0


@POSTPROCESSING_REGISTRY.register("rtmpose_simcc_decode")
class RTMPoseSimCCDecode:
    """Decode SimCC heatmaps to keypoint coordinates and scores.

    Input: list of 2 tensors [simcc_x [1,17,W_bins], simcc_y [1,17,H_bins]]
    Output: (keypoints [17,2], scores [17])
    """

    def __init__(self, split_ratio: float = 2.0, **kwargs):
        self.split_ratio = split_ratio

    def __call__(self, outputs, **kwargs):
        if not isinstance(outputs, (list, tuple)) or len(outputs) < 2:
            return np.zeros((17, 2), dtype=np.float32), np.zeros(17, dtype=np.float32)
        simcc_x = np.asarray(outputs[0])
        simcc_y = np.asarray(outputs[1])
        if simcc_x.ndim == 2:
            simcc_x = simcc_x[np.newaxis]
        if simcc_y.ndim == 2:
            simcc_y = simcc_y[np.newaxis]

        x_idx = simcc_x[0].argmax(axis=-1).astype(np.float32)
        y_idx = simcc_y[0].argmax(axis=-1).astype(np.float32)
        x_score = simcc_x[0].max(axis=-1)
        y_score = simcc_y[0].max(axis=-1)
        scores = (x_score + y_score) / 2.0
        kpts = np.stack([x_idx / self.split_ratio, y_idx / self.split_ratio], axis=-1)
        return kpts, scores
