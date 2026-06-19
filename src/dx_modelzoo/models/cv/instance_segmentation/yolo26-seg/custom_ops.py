"""YOLO26-seg instance segmentation output decoding."""
from __future__ import annotations

import numpy as np

from dx_modelzoo.postprocessing import POSTPROCESSING_REGISTRY
from dx_modelzoo.postprocessing.coord_scaler import unpad_and_scale
from dx_modelzoo.postprocessing.masks import process_masks_fast


@POSTPROCESSING_REGISTRY.register("yolo26_seg_decode")
class YOLO26SegDecode:
    """YOLO26 seg: decode masks from coefficients + proto.

    Input: list of 2 tensors [predictions(1,M,38), proto(1,32,160,160)]
    Output: (detections[M,6], masks[M,origH,origW])
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
        if not isinstance(outputs, (list, tuple)) or len(outputs) < 2:
            return (np.empty((0, 6), dtype=np.float64), None)

        preds = outputs[0]
        proto = outputs[1]
        if preds.ndim == 3:
            preds = preds[0]
        if proto.ndim == 4:
            proto = proto[0]  # (32, 160, 160)

        # Conf threshold
        if preds.shape[0] > 0 and preds.shape[1] >= 5:
            mask = preds[:, 4] > self.conf_thres
            preds = preds[mask]
        if preds.shape[0] > self.max_output_boxes:
            preds = preds[: self.max_output_boxes]

        if preds.shape[0] == 0:
            return (np.empty((0, 6), dtype=np.float64), None)

        boxes = preds[:, :4]  # xyxy in model input space
        detections = preds[:, :6].copy()  # [x1,y1,x2,y2,score,cls]
        mask_coeffs = preds[:, 6:]  # (M, 32)

        # Decode masks using proto (before box rescale — masks use model-space coords)
        origin_hw = kwargs.get("origin_hw")
        input_hw = kwargs.get("input_hw")
        if input_hw is None:
            input_hw = (640, 640)
        if origin_hw is None:
            origin_hw = input_hw

        masks = process_masks_fast(proto, mask_coeffs, boxes, input_hw, origin_hw)

        # Rescale box coordinates to original image space
        detections[:, :4] = unpad_and_scale(detections[:, :4], input_hw, origin_hw, pad_resize=self.pad_resize)

        return (detections, masks)
