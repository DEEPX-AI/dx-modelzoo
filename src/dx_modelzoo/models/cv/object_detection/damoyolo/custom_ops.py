"""DAMO-YOLO output decoding: multi-format handling, boxes in xyxy."""
from __future__ import annotations

import numpy as np

from dx_modelzoo.postprocessing import POSTPROCESSING_REGISTRY
from dx_modelzoo.postprocessing.decode_utils import build_nms_input, split_box_cls_xyxy


@POSTPROCESSING_REGISTRY.register("damoyolo_decode")
class DamoYOLODecode:
    """DAMO-YOLO: handles single/multi output formats.

    Supported:
    - Single [B, N, 4+C]: boxes(4, xyxy) + class_scores(C)
    - Single [B, N, 4+1+C]: boxes(4) + obj(1) + class_scores(C), multiply obj×cls
    - Two outputs: separate boxes(N,4) and scores(N,C)
    """

    def __init__(self, num_class: int = 80, **kwargs):
        self.num_class = num_class

    def __call__(self, outputs, **kwargs):
        if not isinstance(outputs, list):
            outputs = [outputs]

        if len(outputs) == 1:
            out = outputs[0]
            if out.ndim == 3:
                out = out[0]
            last_dim = out.shape[-1]

            if last_dim == self.num_class:
                return build_nms_input(
                    np.empty((0, 4), dtype=np.float64),
                    np.empty((0,), dtype=np.float64),
                    np.empty((0,), dtype=np.float64),
                )

            if last_dim == 4 + 1 + self.num_class:
                # obj_conf × class_scores
                box_data = out[:, :4]
                obj_conf = out[:, 4:5]
                cls_scores = out[:, 5:]
                scores_combined = obj_conf * cls_scores
                out = np.concatenate([box_data, scores_combined], axis=-1)

            boxes, scores, class_ids = split_box_cls_xyxy(out)
            return build_nms_input(boxes, scores, class_ids)

        # Two outputs: identify boxes vs scores by shape
        if len(outputs) == 2:
            b0 = outputs[0][0] if outputs[0].ndim == 3 else outputs[0]
            b1 = outputs[1][0] if outputs[1].ndim == 3 else outputs[1]

            if b0.shape[-1] == 4:
                out = np.concatenate([b0, b1], axis=-1)
            elif b1.shape[-1] == 4:
                out = np.concatenate([b1, b0], axis=-1)
            elif b0.shape[-1] == self.num_class:
                out = np.concatenate([b1, b0], axis=-1)
            else:
                out = np.concatenate([b0, b1], axis=-1)
        else:
            parts = [o[0] if o.ndim == 3 else o for o in outputs]
            out = np.concatenate(parts, axis=-1)

        expected = 4 + self.num_class
        if out.shape[-1] != expected:
            return build_nms_input(
                np.empty((0, 4), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
            )

        boxes, scores, class_ids = split_box_cls_xyxy(out)
        return build_nms_input(boxes, scores, class_ids)
