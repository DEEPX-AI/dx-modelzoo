"""MediaPipePose / BlazePose direct landmark decoder."""
from __future__ import annotations

from typing import List

import numpy as np

from dx_modelzoo.postprocessing import POSTPROCESSING_REGISTRY

_BLAZEPOSE_TO_COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


@POSTPROCESSING_REGISTRY.register("mediapipe_pose_decode")
class MediaPipePoseDecode:
    """Decode direct landmark coordinates (MediaPipePose/BlazePose) to COCO 17-keypoint format.

    Input: list of model outputs [landmarks [1, N*5], optional person_conf [1,1]]
    Output: (keypoints [17,2], scores [17])
    """

    def __init__(self, landmark_mapping: List[int] = None, **kwargs):
        self.landmark_mapping = landmark_mapping or _BLAZEPOSE_TO_COCO

    def __call__(self, outputs, **kwargs):
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]

        raw_landmarks = outputs[0]

        # Person confidence (optional second output)
        person_conf = 1.0
        if len(outputs) >= 2:
            person_conf = float(np.asarray(outputs[1]).flatten()[0])
            if person_conf < 0 or person_conf > 1:
                person_conf = 1.0 / (1.0 + np.exp(-person_conf))

        lm = np.asarray(raw_landmarks).flatten()
        num_lm = len(lm) // 5
        lm = lm.reshape(num_lm, 5)

        kpts_17 = np.zeros((17, 2), dtype=np.float32)
        scores_17 = np.zeros(17, dtype=np.float32)
        for coco_idx, src_idx in enumerate(self.landmark_mapping):
            if src_idx < num_lm:
                kpts_17[coco_idx, 0] = lm[src_idx, 0]
                kpts_17[coco_idx, 1] = lm[src_idx, 1]
                vis = lm[src_idx, 3]
                scores_17[coco_idx] = 1.0 / (1.0 + np.exp(-vis))

        # Modulate by person confidence
        scores_17 = scores_17 * person_conf

        return kpts_17, scores_17
