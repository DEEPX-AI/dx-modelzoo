"""ULFGFD postprocessing custom ops.

Handles two sub-variants:
- ulfgfd: Decoded outputs in normalized coords with direct resize
- ulfgfd_raw: Raw SSD-encoded outputs needing prior-based decoding
"""
from __future__ import annotations

import numpy as np

from dx_modelzoo.postprocessing import POSTPROCESSING_REGISTRY
from dx_modelzoo.postprocessing.box_decoder import generate_ulfgfd_priors
from dx_modelzoo.postprocessing.coord_scaler import scale_direct, xyxy_to_xywh_list
from dx_modelzoo.postprocessing.nms import nms_numpy


@POSTPROCESSING_REGISTRY.register("ulfgfd_decode")
class ULFGFDDecode:
    """Ultra-Light-Fast-Generic-Face-Detector decoder."""

    def __init__(
        self,
        variant: str = "ulfgfd",
        conf_thres: float = 0.5,
        iou_thres: float = 0.3,
        **kwargs,
    ) -> None:
        self.variant = variant
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def __call__(self, outputs, origin_hw=None, input_hw=None, **kwargs):
        if input_hw is None:
            input_hw = origin_hw
        if origin_hw is None:
            origin_hw = input_hw

        if self.variant == "ulfgfd_raw":
            return self._decode_raw(outputs, origin_hw, input_hw)
        return self._decode(outputs, origin_hw, input_hw)

    def _decode(self, outputs, origin_hw, input_hw):
        h_model, w_model = input_hw
        scores_raw, boxes_raw = _identify_scores_boxes(outputs)
        scores = scores_raw[:, 1]
        boxes = boxes_raw.copy()
        boxes[:, [0, 2]] *= w_model
        boxes[:, [1, 3]] *= h_model
        mask = scores > self.conf_thres
        boxes, scores = boxes[mask], scores[mask]
        if len(boxes) == 0:
            return []
        boxes = scale_direct(boxes, input_hw, origin_hw)
        keep = nms_numpy(boxes, scores, self.iou_thres)
        boxes, scores = boxes[keep], scores[keep]
        return xyxy_to_xywh_list(boxes, scores)

    def _decode_raw(self, outputs, origin_hw, input_hw):
        h_model, w_model = input_hw
        scores_raw, loc_raw = _identify_scores_boxes(outputs)
        priors = generate_ulfgfd_priors((h_model, w_model))
        variances = (0.1, 0.2)
        cx = priors[:, 0] + loc_raw[:, 0] * variances[0] * priors[:, 2]
        cy = priors[:, 1] + loc_raw[:, 1] * variances[0] * priors[:, 3]
        w = priors[:, 2] * np.exp(np.clip(loc_raw[:, 2] * variances[1], -50.0, 50.0))
        h = priors[:, 3] * np.exp(np.clip(loc_raw[:, 3] * variances[1], -50.0, 50.0))
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
        boxes[:, [0, 2]] *= w_model
        boxes[:, [1, 3]] *= h_model
        scores = scores_raw[:, 1]
        mask = scores > self.conf_thres
        boxes, scores = boxes[mask], scores[mask]
        if len(boxes) == 0:
            return []
        boxes = scale_direct(boxes, input_hw, origin_hw)
        keep = nms_numpy(boxes, scores, self.iou_thres)
        boxes, scores = boxes[keep], scores[keep]
        return xyxy_to_xywh_list(boxes, scores)


def _identify_scores_boxes(outputs):
    loc, conf = None, None
    for out in outputs:
        last_dim = out.shape[-1]
        if last_dim == 4:
            loc = np.squeeze(out)
        elif last_dim == 2:
            conf = np.squeeze(out)
    if loc is None or conf is None:
        return np.empty((0, 2)), np.empty((0, 4))
    return conf, loc
