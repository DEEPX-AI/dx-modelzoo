"""YOLOv7 output decoding: anchor-based multi-scale or pre-decoded."""
from __future__ import annotations

import numpy as np

from dx_modelzoo.postprocessing import POSTPROCESSING_REGISTRY
from dx_modelzoo.postprocessing.decode_utils import (
    YOLOV7_ANCHORS,
    YOLOV7_STRIDES,
    apply_obj_cls_score,
    build_nms_input,
    sigmoid,
)


@POSTPROCESSING_REGISTRY.register("yolov7_decode")
class YOLOv7Decode:
    """YOLOv7 anchor-based multi-scale decode or pre-decoded passthrough."""

    def __init__(self, variant: str = "yolov7", num_class: int = 80, **kwargs):
        self.variant = variant  # "yolov7" or "yolov7_w6"
        self.num_class = num_class

    def __call__(self, outputs, **kwargs):
        if not isinstance(outputs, list):
            outputs = [outputs]

        # Already decoded: single [1, N, 85]
        if len(outputs) == 1 and outputs[0].ndim == 3 and outputs[0].shape[-1] == 5 + self.num_class:
            decoded = outputs[0][0]  # [N, 5+C]
            boxes, scores, class_ids = apply_obj_cls_score(decoded)
            return build_nms_input(boxes, scores, class_ids)

        # Multi-scale raw decode
        decoded = self._decode_multiscale(outputs)
        out = decoded[0]  # remove batch → [N, 5+C]
        boxes, scores, class_ids = apply_obj_cls_score(out)
        return build_nms_input(boxes, scores, class_ids)

    def _decode_multiscale(self, outputs):
        anchors = YOLOV7_ANCHORS[self.variant]
        strides = YOLOV7_STRIDES
        num_scales = len(outputs)
        anchors = anchors[:num_scales]
        strides = strides[:num_scales]

        all_preds = []
        for scale_idx, output in enumerate(outputs):
            if output.ndim != 5:
                continue
            batch, na, h, w, nc = output.shape
            stride = strides[scale_idx]
            xv, yv = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
            grid = np.stack((xv, yv), axis=2).reshape(1, 1, h, w, 2).astype(np.float32)
            anchor_grid = np.array(anchors[scale_idx], dtype=np.float32).reshape(1, na, 1, 1, 2)
            y = sigmoid(output)
            xy = (y[..., 0:2] * 2.0 - 0.5 + grid) * stride
            wh = (y[..., 2:4] * 2.0) ** 2 * anchor_grid
            obj_conf = y[..., 4:5]
            cls_conf = y[..., 5:]
            pred = np.concatenate([xy, wh, obj_conf, cls_conf], axis=-1)
            all_preds.append(pred.reshape(batch, -1, nc))

        if not all_preds:
            return np.empty((1, 0, 5 + self.num_class), dtype=np.float32)
        return np.concatenate(all_preds, axis=1)
