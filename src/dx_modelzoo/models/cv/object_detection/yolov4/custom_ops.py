"""YOLOv4 output decoding: separate boxes + class scores."""
from __future__ import annotations

import numpy as np

from dx_modelzoo.postprocessing import POSTPROCESSING_REGISTRY
from dx_modelzoo.postprocessing.decode_utils import build_nms_input


@POSTPROCESSING_REGISTRY.register("yolov4_decode")
class YOLOv4Decode:
    """YOLOv4: separate boxes [B,N,1,4] + scores [B,N,C], normalized coords."""

    def __init__(self, input_size: int = 416, **kwargs):
        self.input_size = input_size

    def __call__(self, outputs, **kwargs):
        if not isinstance(outputs, list) or len(outputs) < 2:
            return np.empty((0, 6), dtype=np.float64)

        boxes = outputs[0]
        confs = outputs[1]

        if boxes.ndim == 4:
            boxes = boxes.squeeze(2)  # [B, N, 1, 4] → [B, N, 4]
        if boxes.ndim == 2:
            boxes = boxes[np.newaxis, ...]
        if confs.ndim == 2:
            confs = confs[np.newaxis, ...]

        # Scale normalized to pixel coords
        boxes = boxes[0] * self.input_size  # [N, 4] xyxy
        confs = confs[0]  # [N, C]

        scores = confs.max(axis=1)
        class_ids = confs.argmax(axis=1).astype(np.float64)

        return build_nms_input(boxes.copy(), scores, class_ids)
