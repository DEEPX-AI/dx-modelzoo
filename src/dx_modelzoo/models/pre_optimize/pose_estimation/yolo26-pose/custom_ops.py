"""YOLO26-pose pose estimation output decoding."""
from __future__ import annotations

import numpy as np

from dx_modelzoo.postprocessing import POSTPROCESSING_REGISTRY
from dx_modelzoo.postprocessing.coord_scaler import unpad_and_scale


@POSTPROCESSING_REGISTRY.register("yolo26_pose_decode")
class YOLO26PoseDecode:
    """YOLO26 pose: split decoded output into (boxes, scores, cls, keypoints).

    Input: [M, 6+K*3] ndarray (xyxy, score, cls, keypoints...)
    Output: (boxes[M,4], scores[M], cls[M], keypoints[M,K,3])
    """

    def __init__(
        self,
        num_keypoints: int = 17,
        conf_thres: float = 0.001,
        max_output_boxes: int = 300,
        pad_resize: bool = True,
        **kwargs,
    ):
        self.num_keypoints = num_keypoints
        self.conf_thres = conf_thres
        self.max_output_boxes = max_output_boxes
        self.pad_resize = pad_resize

    def __call__(self, outputs, **kwargs):
        out = outputs[0] if isinstance(outputs, list) else outputs
        if out.ndim == 3:
            out = out[0]

        # Conf threshold
        if out.shape[0] > 0 and out.shape[1] >= 5:
            mask = out[:, 4] > self.conf_thres
            out = out[mask]
        if out.shape[0] > self.max_output_boxes:
            out = out[: self.max_output_boxes]

        if out.shape[0] == 0:
            empty = np.empty((0, 4), dtype=np.float64)
            return (empty, np.empty(0), np.empty(0), np.empty((0, self.num_keypoints, 3)))

        boxes = out[:, :4]
        scores = out[:, 4]
        cls = out[:, 5]
        kpt_data = out[:, 6:]
        keypoints = kpt_data.reshape(-1, self.num_keypoints, 3)

        # Rescale boxes and keypoints to original image space
        origin_hw = kwargs.get("origin_hw")
        input_hw = kwargs.get("input_hw")
        if origin_hw is not None and input_hw is not None:
            boxes = unpad_and_scale(boxes, input_hw, origin_hw, pad_resize=self.pad_resize)
            # Rescale keypoint x,y (columns 0,1 of each keypoint)
            h_model, w_model = input_hw
            h_orig, w_orig = origin_hw
            if self.pad_resize:
                ratio = min(h_model / h_orig, w_model / w_orig)
                pad_w = (w_model - w_orig * ratio) / 2
                pad_h = (h_model - h_orig * ratio) / 2
                keypoints[:, :, 0] = (keypoints[:, :, 0] - pad_w) / ratio
                keypoints[:, :, 1] = (keypoints[:, :, 1] - pad_h) / ratio
            else:
                keypoints[:, :, 0] *= w_orig / w_model
                keypoints[:, :, 1] *= h_orig / h_model
            keypoints[:, :, 0] = np.clip(keypoints[:, :, 0], 0, w_orig)
            keypoints[:, :, 1] = np.clip(keypoints[:, :, 1], 0, h_orig)

        return (boxes, scores, cls, keypoints)
