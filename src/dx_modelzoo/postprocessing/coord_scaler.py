"""Coordinate scaling utilities for detection postprocessing.

Handles reverse pad-resize, direct-resize scaling, box format conversions.
All pure numpy, no torch.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def unpad_and_scale(
    boxes: np.ndarray,
    model_hw: Tuple[int, int],
    orig_hw: Tuple[int, int],
    pad_resize: bool = True,
) -> np.ndarray:
    """Reverse pad-resize or direct-resize to map boxes to original image coords.

    Args:
        boxes: [N, 4] boxes in xyxy format, in model input coordinate space.
        model_hw: (height, width) of the model input.
        orig_hw: (height, width) of the original image.
        pad_resize: If True, reverse pad-resize (letterbox). If False, direct resize.

    Returns:
        [N, 4] boxes in xyxy format, clipped to original image bounds.
    """
    h_model, w_model = model_hw
    h_orig, w_orig = orig_hw
    boxes = boxes.copy().astype(np.float64)

    if pad_resize:
        ratio = min(h_model / h_orig, w_model / w_orig)
        pad_w = (w_model - w_orig * ratio) / 2
        pad_h = (h_model - h_orig * ratio) / 2
        boxes[:, [0, 2]] -= pad_w
        boxes[:, [1, 3]] -= pad_h
        boxes[:, :4] /= ratio
    else:
        boxes[:, [0, 2]] *= w_orig / w_model
        boxes[:, [1, 3]] *= h_orig / h_model

    boxes[:, 0] = np.clip(boxes[:, 0], 0, w_orig)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h_orig)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w_orig)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h_orig)
    return boxes


def scale_direct(
    boxes: np.ndarray,
    model_hw: Tuple[int, int],
    orig_hw: Tuple[int, int],
) -> np.ndarray:
    """Direct resize scaling (no padding). Shortcut for unpad_and_scale(pad_resize=False)."""
    return unpad_and_scale(boxes, model_hw, orig_hw, pad_resize=False)


def xyxy_to_xywh_list(
    boxes: np.ndarray,
    scores: np.ndarray,
) -> List[List[float]]:
    """Convert [N, 4] xyxy boxes + [N] scores to list of [x, y, w, h, conf].

    Used for WiderFace evaluation format.
    """
    result = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        result.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1), float(scores[i])])
    return result
