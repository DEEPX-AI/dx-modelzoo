"""Semantic segmentation argmax postprocessing.

Handles NCHW and NHWC layouts, single-channel squeeze, and optional resize.
Covers: deeplab, bisenet, unet, fcn, segformer models.
"""
from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from dx_modelzoo.postprocessing import POSTPROCESSING_REGISTRY


@POSTPROCESSING_REGISTRY.register("segmentation_argmax")
class SegmentationArgmax:
    """Argmax on class dimension + optional resize to target size.

    Args:
        layout: Input layout, "nchw" or "nhwc". Default "nchw".
        target_size: Optional [H, W] to resize output via nearest interpolation.
    """

    def __init__(
        self,
        layout: str = "nchw",
        target_size: Optional[List[int]] = None,
        **kwargs,
    ) -> None:
        self.layout = layout.lower()
        self.target_size = target_size

    def __call__(self, outputs, **kwargs):
        pred = outputs[0] if isinstance(outputs, list) else outputs

        if pred.ndim == 4:
            if self.layout == "nchw":
                if pred.shape[1] == 1:
                    pred = pred[:, 0, :, :]
                else:
                    pred = np.argmax(pred, axis=1)
            else:  # nhwc
                if pred.shape[-1] == 1:
                    pred = pred[..., 0]
                else:
                    pred = np.argmax(pred, axis=-1)

        if self.target_size is not None:
            h, w = self.target_size[0], self.target_size[1]
            resized = np.zeros((pred.shape[0], h, w), dtype=np.int64)
            for i in range(pred.shape[0]):
                resized[i] = cv2.resize(
                    pred[i].astype(np.uint8),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                )
            pred = resized

        return pred
