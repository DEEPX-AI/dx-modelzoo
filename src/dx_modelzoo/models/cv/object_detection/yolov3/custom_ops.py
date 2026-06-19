"""YOLOv3 output decoding variants."""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from dx_modelzoo.postprocessing import POSTPROCESSING_REGISTRY
from dx_modelzoo.postprocessing.decode_utils import apply_obj_cls_score, build_nms_input, sigmoid


@POSTPROCESSING_REGISTRY.register("yolo_decode")
class YOLODecode:
    """Default YOLO decode: [1, N, 5+C] → obj_conf × cls → NMS input.

    Shared by YOLOv3, YOLOv5, YOLOv6 and other models using variant=yolo.
    """

    def __call__(self, outputs, **kwargs):
        out = outputs[0] if isinstance(outputs, list) else outputs
        if out.ndim == 2:
            out = out[None, ...]
        out = out[0]  # [N, 5+C]
        boxes, scores, class_ids = apply_obj_cls_score(out)
        return build_nms_input(boxes, scores, class_ids)


@POSTPROCESSING_REGISTRY.register("yolov3_gluon_decode")
class YOLOv3GluonDecode:
    """YOLOv3 Gluon multi-scale anchor decode: [B, H, W, 255] per scale."""

    def __init__(
        self,
        anchors: Optional[List] = None,
        strides: Optional[List[int]] = None,
        num_class: int = 80,
        **kwargs,
    ):
        self.anchors = anchors or [
            [[116, 90], [156, 198], [373, 326]],
            [[30, 61], [62, 45], [59, 119]],
            [[10, 13], [16, 30], [33, 23]],
        ]
        self.strides = strides or [32, 16, 8]
        self.num_class = num_class

    def __call__(self, outputs, **kwargs):
        if not isinstance(outputs, list):
            outputs = [outputs]

        # If already decoded [1, N, 85], use simple decode
        if len(outputs) == 1 and outputs[0].ndim == 3 and outputs[0].shape[-1] == 5 + self.num_class:
            out = outputs[0][0]
            boxes, scores, class_ids = apply_obj_cls_score(out)
            return build_nms_input(boxes, scores, class_ids)

        num_scales = min(len(outputs), len(self.anchors), len(self.strides))
        all_preds = []

        for scale_idx in range(num_scales):
            output = outputs[scale_idx].copy().astype(np.float32)
            stride = self.strides[scale_idx]
            anchor = np.array(self.anchors[scale_idx], dtype=np.float32)
            na = len(anchor)

            # [B, H, W, 255] → [B, H, W, 3, 85]
            if output.ndim == 3:
                output = output[np.newaxis, ...]
            B, H, W, C = output.shape
            nc = C // na
            output = output.reshape(B, H, W, na, nc)

            xv, yv = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
            grid = np.stack((xv, yv), axis=2).reshape(1, H, W, 1, 2).astype(np.float32)
            anchor_grid = anchor.reshape(1, 1, 1, na, 2)

            y = sigmoid(output)
            # YOLOv3 Gluon: xy = (sigmoid(xy) + grid) * stride, wh = exp(raw_wh) * anchor
            xy = (y[..., :2] + grid) * stride
            wh = np.exp(output[..., 2:4]) * anchor_grid
            obj_conf = y[..., 4:5]
            cls_conf = y[..., 5:]

            pred = np.concatenate([xy, wh, obj_conf, cls_conf], axis=-1)
            all_preds.append(pred.reshape(B, -1, nc))

        if not all_preds:
            return build_nms_input(
                np.empty((0, 4), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
            )

        decoded = np.concatenate(all_preds, axis=1)[0]
        boxes, scores, class_ids = apply_obj_cls_score(decoded)
        return build_nms_input(boxes, scores, class_ids)
