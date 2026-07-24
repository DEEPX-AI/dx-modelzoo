"""RetinaFace postprocessing custom ops.

Handles two sub-variants:
- retinaface: SSD-style decoded outputs with pad-resize
- retinaface_v1: Per-level spatial outputs (NHWC) with direct resize
"""
from __future__ import annotations

import numpy as np

from dx_modelzoo.postprocessing import POSTPROCESSING_REGISTRY
from dx_modelzoo.postprocessing.box_decoder import decode_ssd_boxes, generate_ssd_priors
from dx_modelzoo.postprocessing.coord_scaler import scale_direct, unpad_and_scale, xyxy_to_xywh_list
from dx_modelzoo.postprocessing.nms import nms_numpy


@POSTPROCESSING_REGISTRY.register("retinaface_decode")
class RetinaFaceDecode:
    """RetinaFace face detection decoder."""

    def __init__(
        self,
        variant: str = "retinaface",
        conf_thres: float = 0.02,
        iou_thres: float = 0.4,
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

        if self.variant == "retinaface_v1":
            return self._decode_v1(outputs, origin_hw, input_hw)
        return self._decode(outputs, origin_hw, input_hw)

    def _decode(self, outputs, origin_hw, input_hw):
        h_model, w_model = input_hw
        loc, conf = _identify_loc_conf(outputs)
        if loc.shape[0] == 0:
            return []
        priors = generate_ssd_priors((h_model, w_model))
        n = min(loc.shape[0], priors.shape[0])
        loc, priors, conf = loc[:n], priors[:n], conf[:n]
        boxes = decode_ssd_boxes(loc, priors)
        boxes *= np.array([w_model, h_model, w_model, h_model])
        scores = conf[:, 1]
        boxes, scores = _filter_conf(boxes, scores, self.conf_thres)
        if len(boxes) == 0:
            return []
        boxes = unpad_and_scale(boxes, input_hw, origin_hw, pad_resize=True)
        boxes, scores = _apply_nms(boxes, scores, self.iou_thres)
        return xyxy_to_xywh_list(boxes, scores)

    def _decode_v1(self, outputs, origin_hw, input_hw):
        h_model, w_model = input_hw
        n_anchors = 2
        loc_list, conf_list = [], []
        for out in outputs:
            out = np.squeeze(out, axis=0)
            last_dim = out.shape[-1]
            per_anchor = last_dim // n_anchors
            reshaped = out.reshape(-1, per_anchor)
            if per_anchor == 4:
                loc_list.append(reshaped)
            elif per_anchor == 2:
                conf_list.append(reshaped)
        if not loc_list or not conf_list:
            return []
        loc = np.concatenate(loc_list, axis=0)
        conf = np.concatenate(conf_list, axis=0)
        priors = generate_ssd_priors((h_model, w_model))
        boxes = decode_ssd_boxes(loc, priors)
        boxes *= np.array([w_model, h_model, w_model, h_model])
        scores = conf[:, 1]
        boxes, scores = _filter_conf(boxes, scores, self.conf_thres)
        if len(boxes) == 0:
            return []
        boxes = scale_direct(boxes, input_hw, origin_hw)
        boxes, scores = _apply_nms(boxes, scores, self.iou_thres)
        return xyxy_to_xywh_list(boxes, scores)


def _identify_loc_conf(outputs):
    loc, conf = None, None
    for out in outputs:
        last_dim = out.shape[-1]
        if last_dim == 4:
            loc = np.squeeze(out)
        elif last_dim == 2:
            conf = np.squeeze(out)
    if loc is None or conf is None:
        return np.empty((0, 4)), np.empty((0, 2))
    return loc, conf


def _filter_conf(boxes, scores, conf_thres):
    mask = scores > conf_thres
    return boxes[mask], scores[mask]


def _apply_nms(boxes, scores, iou_thres):
    if len(boxes) == 0:
        return boxes, scores
    keep = nms_numpy(boxes, scores, iou_thres)
    return boxes[keep], scores[keep]
