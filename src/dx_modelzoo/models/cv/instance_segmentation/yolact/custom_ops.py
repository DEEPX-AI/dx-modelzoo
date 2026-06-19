"""YOLACT instance segmentation postprocessing custom ops."""
from __future__ import annotations

import math

import numpy as np

from dx_modelzoo.postprocessing import POSTPROCESSING_REGISTRY
from dx_modelzoo.postprocessing.coord_scaler import unpad_and_scale
from dx_modelzoo.postprocessing.decode_utils import cxcywh_to_xyxy
from dx_modelzoo.postprocessing.masks import process_masks, process_masks_fast
from dx_modelzoo.postprocessing.nms import nms_numpy


def _make_yolact_priors(input_size: int) -> np.ndarray:
    """Generate YOLACT priors as normalized (cx, cy, w, h)."""
    feature_sizes = [input_size // stride for stride in (8, 16, 32, 64, 128)]
    base_scales = [24.0, 48.0, 96.0, 192.0, 384.0]
    scale_factor = input_size / 550.0
    octave_scales = [1.0, 2 ** (1 / 3), 2 ** (2 / 3)]
    aspect_ratios = [1.0, 0.5, 2.0]

    priors = []
    for feature_size, base_scale in zip(feature_sizes, base_scales):
        scaled_base = base_scale * scale_factor
        for y in range(feature_size):
            cy = (y + 0.5) / feature_size
            for x in range(feature_size):
                cx = (x + 0.5) / feature_size
                for octave_scale in octave_scales:
                    scale = scaled_base * octave_scale
                    for aspect_ratio in aspect_ratios:
                        w = scale * math.sqrt(aspect_ratio) / input_size
                        h = scale / math.sqrt(aspect_ratio) / input_size
                        priors.append([cx, cy, w, h])
    return np.asarray(priors, dtype=np.float32)


def _decode_yolact_boxes(loc: np.ndarray, input_size: int) -> np.ndarray:
    """Decode YOLACT loc offsets against default priors into xyxy pixel boxes."""
    priors = _make_yolact_priors(input_size)
    if priors.shape[0] != loc.shape[0]:
        return np.clip(loc, 0.0, 1.0) * input_size

    center = priors[:, :2] + loc[:, :2] * 0.1 * priors[:, 2:]
    size = priors[:, 2:] * np.exp(np.clip(loc[:, 2:] * 0.2, -50, 50))
    boxes = cxcywh_to_xyxy(np.concatenate([center, size], axis=1))
    return np.clip(boxes, 0.0, 1.0) * input_size


@POSTPROCESSING_REGISTRY.register("yolact_decode")
class YolactDecode:
    """YOLACT postprocessing: top-k + batched NMS + mask coefficients.

    Expected outputs: [loc, conf, mask_coeff, proto] (4 or 5 elements).
    Returns: (detections(M,6), masks(M,origH,origW)) tuple.
    """

    def __init__(
        self,
        conf_thres: float = 0.001,
        iou_thres: float = 0.7,
        max_det: int = 300,
        pad_resize: bool = True,
        **kwargs,
    ) -> None:
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.pad_resize = pad_resize

    def __call__(self, outputs, **kwargs):
        if len(outputs) < 4:
            return np.empty((0, 6), dtype=np.float32), None

        if len(outputs) == 5:
            loc, conf, mask_coeff, _, proto = outputs
        else:
            loc, conf, mask_coeff, proto = outputs

        loc = loc[0] if loc.ndim == 3 else loc
        conf = conf[0] if conf.ndim == 3 else conf
        mask_coeff = mask_coeff[0] if mask_coeff.ndim == 3 else mask_coeff

        if proto.ndim == 4:
            proto = proto[0]
        if proto.ndim == 3 and proto.shape[-1] == 32:
            proto = proto.transpose(2, 0, 1)

        input_size = proto.shape[1] * 4

        conf_fg = conf[:, 1:]
        max_scores = conf_fg.max(axis=1)
        class_ids = conf_fg.argmax(axis=1)

        keep_mask = max_scores > self.conf_thres
        if not keep_mask.any():
            return np.empty((0, 6), dtype=np.float32), None

        keep_indices = np.nonzero(keep_mask)[0]

        top_k = self.max_det
        if len(keep_indices) > top_k:
            topk_idx = np.argpartition(max_scores[keep_indices], -top_k)[-top_k:]
            keep_indices = keep_indices[topk_idx]

        decoded_boxes = _decode_yolact_boxes(loc, input_size)
        k_scores = max_scores[keep_indices]
        k_class_ids = class_ids[keep_indices]
        k_boxes = decoded_boxes[keep_indices]
        k_masks = mask_coeff[keep_indices]

        offsets = k_class_ids[:, None].astype(np.float32) * (input_size + 1)
        boxes_for_nms = k_boxes + offsets
        keep = nms_numpy(boxes_for_nms, k_scores, self.iou_thres)

        if len(keep) == 0:
            return np.empty((0, 6), dtype=np.float32), None

        keep = keep[: self.max_det]
        det_boxes = k_boxes[keep]
        det_scores = k_scores[keep]
        det_cls = k_class_ids[keep].astype(np.float32)
        det_mask_coeffs = k_masks[keep]

        order = np.argsort(-det_scores, kind="stable")
        det_boxes = det_boxes[order]
        det_scores = det_scores[order]
        det_cls = det_cls[order]
        det_mask_coeffs = det_mask_coeffs[order]

        origin_hw = kwargs.get("origin_hw")
        input_hw = kwargs.get("input_hw")
        final_masks = None
        if origin_hw is not None and input_hw is not None:
            if proto.shape[0] == det_mask_coeffs.shape[1]:
                final_masks = process_masks_fast(proto, det_mask_coeffs, det_boxes, input_hw, origin_hw)
            else:
                final_masks = process_masks(proto, det_mask_coeffs, det_boxes, input_hw, origin_hw)

        if origin_hw is not None and input_hw is not None:
            det_boxes = unpad_and_scale(det_boxes.copy(), input_hw, origin_hw, pad_resize=self.pad_resize)

        result = np.column_stack([det_boxes, det_scores, det_cls]).astype(np.float32)
        return result, final_masks
