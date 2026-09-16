"""Mask RT-DETR output normalization for COCO evaluation."""
from __future__ import annotations

import cv2
import numpy as np

from dx_modelzoo.postprocessing import POSTPROCESSING_REGISTRY
from dx_modelzoo.postprocessing.rtdetr import decode_rows


@POSTPROCESSING_REGISTRY.register("mask_rtdetr_decode")
class MaskRTDETRDecode:
    """Convert ``[detections, count, masks]`` into evaluator-ready outputs."""

    def __init__(self, pad_location: str = "edge", **kwargs) -> None:
        self.pad_location = pad_location

    def __call__(self, outputs, **kwargs):
        if len(outputs) == 3:
            detections, count, masks = outputs
            count = int(np.asarray(count).reshape(-1)[0])
        else:
            detections, masks = outputs
            count = len(detections)
        detections = np.asarray(detections)[:count].astype(np.float64, copy=False)
        masks = np.asarray(masks)[:count].astype(np.uint8, copy=False)
        decoded = decode_rows(
            (detections, np.asarray([count])),
            input_hw=kwargs.get("input_hw"),
            origin_hw=kwargs.get("origin_hw"),
            pad_location=self.pad_location,
        )

        input_hw = kwargs.get("input_hw")
        origin_hw = kwargs.get("origin_hw")
        if input_hw is not None and origin_hw is not None:
            h_orig, w_orig = origin_hw
            h_model, w_model = input_hw
            ratio = min(h_model / h_orig, w_model / w_orig)
            resized_h = int(round(h_orig * ratio))
            resized_w = int(round(w_orig * ratio))
            if self.pad_location == "direct":
                cropped = masks
            else:
                pad_h = 0 if self.pad_location == "back" else int(round((h_model - resized_h) / 2))
                pad_w = 0 if self.pad_location == "back" else int(round((w_model - resized_w) / 2))
                cropped = masks[:, pad_h : pad_h + resized_h, pad_w : pad_w + resized_w]
            if len(masks):
                # ponytail: official DETRPostProcess bilinear-upsamples mask logits to the
                # original resolution before thresholding. The exported ONNX already
                # binarised at model resolution, so bilinear + 0.5 threshold is the closest
                # approximation; nearest costs ~1p mask AP.
                masks = np.stack(
                    [
                        (
                            cv2.resize(mask.astype(np.float32), (w_orig, h_orig), interpolation=cv2.INTER_LINEAR) >= 0.5
                        ).astype(np.uint8)
                        for mask in cropped
                    ],
                    axis=0,
                )
            else:
                masks = np.empty((0, h_orig, w_orig), dtype=np.uint8)

        return decoded, masks
