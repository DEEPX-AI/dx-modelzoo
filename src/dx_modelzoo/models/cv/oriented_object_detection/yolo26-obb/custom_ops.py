"""YOLO26-OBB oriented object detection output decoding."""
from __future__ import annotations

import numpy as np

from dx_modelzoo.postprocessing import POSTPROCESSING_REGISTRY


@POSTPROCESSING_REGISTRY.register("yolo26_obb_decode")
class YOLO26OBBDecode:
    """YOLO26 OBB: decode oriented bounding boxes.

    Input: [M, 7] ndarray (cx, cy, w, h, score, cls, angle)
    Output: dict(boxes[M,4], scores[M], labels[M], angles[M])
    """

    def __init__(
        self,
        conf_thres: float = 0.001,
        max_output_boxes: int = 300,
        pad_resize: bool = True,
        **kwargs,
    ):
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
            return {
                "boxes": np.zeros((0, 4), dtype=np.float64),
                "scores": np.zeros(0, dtype=np.float64),
                "labels": np.zeros(0, dtype=np.float64),
                "angles": np.zeros(0, dtype=np.float64),
            }

        # [cx, cy, w, h, score, cls, angle]
        cx = out[:, 0].copy()
        cy = out[:, 1].copy()
        w = out[:, 2].copy()
        h = out[:, 3].copy()
        scores = out[:, 4]
        labels = out[:, 5]
        angles = out[:, 6]

        # Rescale cx, cy, w, h to original image space
        origin_hw = kwargs.get("origin_hw")
        input_hw = kwargs.get("input_hw")
        if origin_hw is not None and input_hw is not None:
            h_model, w_model = input_hw
            h_orig, w_orig = origin_hw
            if self.pad_resize:
                ratio = min(h_model / h_orig, w_model / w_orig)
                pad_w = (w_model - w_orig * ratio) / 2
                pad_h = (h_model - h_orig * ratio) / 2
                cx = (cx - pad_w) / ratio
                cy = (cy - pad_h) / ratio
                w = w / ratio
                h = h / ratio
            else:
                cx *= w_orig / w_model
                cy *= h_orig / h_model
                w *= w_orig / w_model
                h *= h_orig / h_model
            cx = np.clip(cx, 0, w_orig)
            cy = np.clip(cy, 0, h_orig)

        boxes = np.stack([cx, cy, w, h], axis=1)
        return {
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
            "angles": angles,
        }
