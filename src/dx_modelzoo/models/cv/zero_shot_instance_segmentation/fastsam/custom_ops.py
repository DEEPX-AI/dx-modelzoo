"""FastSAM zero-shot instance segmentation postprocessing."""
from __future__ import annotations

import numpy as np

from dx_modelzoo.postprocessing import POSTPROCESSING_REGISTRY
from dx_modelzoo.postprocessing.coord_scaler import unpad_and_scale
from dx_modelzoo.postprocessing.decode_utils import cxcywh_to_xyxy
from dx_modelzoo.postprocessing.masks import process_masks, process_masks_fast
from dx_modelzoo.postprocessing.nms import nms_numpy


@POSTPROCESSING_REGISTRY.register("fastsam_decode")
class FastSAMDecode:
    """FastSAM/YOLOv8-seg postprocessing with mask decoding.

    Expected model outputs: [predictions(1,C,N), prototypes(1,32,H,W)] or single tensor.
    Returns: (detections(M,6), masks(M,origH,origW)) tuple.
    """

    def __init__(
        self,
        conf_thres: float = 0.001,
        iou_thres: float = 0.7,
        max_det: int = 300,
        has_objectness: bool = False,
        pad_resize: bool = True,
        **kwargs,
    ) -> None:
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.has_objectness = has_objectness
        self.pad_resize = pad_resize

    def __call__(self, outputs, **kwargs):
        if isinstance(outputs, (list, tuple)) and len(outputs) >= 2:
            raw = outputs[0]
            proto = outputs[1]
        else:
            raw = outputs if not isinstance(outputs, (list, tuple)) else outputs[0]
            proto = None

        if raw.ndim == 3:
            min_channels = 37 if self.has_objectness else 36
            dim1, dim2 = raw.shape[1], raw.shape[2]
            if (dim1 >= min_channels and dim2 < min_channels) or (
                dim1 >= min_channels and dim2 >= min_channels and dim1 < dim2
            ):
                raw = np.transpose(raw, (0, 2, 1))
            raw = raw[0]

        if self.has_objectness:
            nc = raw.shape[1] - 5 - 32
            if nc < 1:
                nc = 1
                class_scores = raw[:, 4:5]
            else:
                mi = 5 + nc
                class_scores = raw[:, 5:mi] * raw[:, 4:5]
        else:
            nc = raw.shape[1] - 4 - 32
            if nc < 1:
                nc = 1
            mi = 4 + nc
            class_scores = raw[:, 4:mi]

        max_conf = class_scores.max(axis=1)
        mask = max_conf > self.conf_thres
        x = raw[mask]

        if x.shape[0] == 0:
            return np.empty((0, 6), dtype=np.float64), None

        boxes = cxcywh_to_xyxy(x[:, :4])
        if self.has_objectness and nc >= 1:
            if raw.shape[1] - 5 - 32 < 1:
                conf = x[:, 4:5].max(axis=1)
                cls_ids = np.zeros(len(x), dtype=np.int64)
                mask_data = x[:, 5:]
            else:
                cs = x[:, 5 : 5 + nc] * x[:, 4:5]
                conf = cs.max(axis=1)
                cls_ids = cs.argmax(axis=1)
                mask_data = x[:, 5 + nc :]
        else:
            conf = x[:, 4 : 4 + nc].max(axis=1)
            cls_ids = x[:, 4 : 4 + nc].argmax(axis=1)
            mask_data = x[:, 4 + nc :]

        order = np.argsort(-conf, kind="stable")[:30000]
        boxes, conf, cls_ids, mask_data = (
            boxes[order],
            conf[order],
            cls_ids[order],
            mask_data[order],
        )

        keep = nms_numpy(boxes, conf, self.iou_thres)
        keep = keep[: self.max_det]

        det_boxes = boxes[keep]
        det_conf = conf[keep]
        det_cls = cls_ids[keep].astype(np.float64)
        det_masks = mask_data[keep]

        origin_hw = kwargs.get("origin_hw")
        input_hw = kwargs.get("input_hw")
        final_masks = None
        if proto is not None and origin_hw is not None and input_hw is not None:
            p = proto[0] if proto.ndim == 4 else proto
            if p.ndim == 3 and p.shape[-1] == 32:
                p = p.transpose(2, 0, 1)
            if p.shape[0] == det_masks.shape[1]:
                final_masks = process_masks_fast(p, det_masks, det_boxes, input_hw, origin_hw)
            else:
                final_masks = process_masks(p, det_masks, det_boxes, input_hw, origin_hw)

        if origin_hw is not None and input_hw is not None:
            det_boxes = unpad_and_scale(det_boxes.copy(), input_hw, origin_hw, pad_resize=self.pad_resize)

        result = np.column_stack([det_boxes, det_conf, det_cls])
        return result, final_masks
