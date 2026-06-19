"""Common mask utilities for instance segmentation postprocessing.

Used by YOLOv5, YOLOv8, YOLACT, Yolo26, and evaluators.
"""
from __future__ import annotations

import math
from typing import Tuple

import cv2
import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def crop_mask(masks: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Crop masks by bounding boxes (numpy version, per-mask slicing)."""
    n, h, w = masks.shape
    result = np.zeros_like(masks)
    for i in range(n):
        x1 = max(0, math.ceil(boxes[i, 0]))
        y1 = max(0, math.ceil(boxes[i, 1]))
        x2 = min(w, math.ceil(boxes[i, 2]))
        y2 = min(h, math.ceil(boxes[i, 3]))
        if x2 > x1 and y2 > y1:
            result[i, y1:y2, x1:x2] = masks[i, y1:y2, x1:x2]
    return result


def process_masks(
    protos: np.ndarray,
    masks_in: np.ndarray,
    bboxes: np.ndarray,
    input_shape: Tuple[int, int],
    original_shape: Tuple[int, int],
) -> np.ndarray:
    """Standard mask processing: resize to input → crop → resize to original."""
    c, mh, mw = protos.shape
    ih, iw = input_shape
    h_orig, w_orig = original_shape

    masks = _sigmoid(masks_in @ protos.reshape(c, -1)).reshape(-1, mh, mw)
    resized = np.empty((len(masks), ih, iw), dtype=np.float32)
    for i in range(len(masks)):
        resized[i] = cv2.resize(masks[i], (iw, ih), interpolation=cv2.INTER_LINEAR)
    masks = resized

    masks = crop_mask(masks, bboxes)

    gain = min(ih / h_orig, iw / w_orig)
    pad_x = (iw - w_orig * gain) / 2
    pad_y = (ih - h_orig * gain) / 2
    top, left = int(round(pad_y)), int(round(pad_x))
    bottom, right = int(round(ih - pad_y)), int(round(iw - pad_x))
    masks = masks[:, top:bottom, left:right]

    final = np.empty((len(masks), h_orig, w_orig), dtype=np.uint8)
    for i in range(len(masks)):
        final[i] = (cv2.resize(masks[i], (w_orig, h_orig), interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.uint8)
    return final


def process_masks_fast(
    protos: np.ndarray,
    masks_in: np.ndarray,
    bboxes: np.ndarray,
    input_shape: Tuple[int, int],
    original_shape: Tuple[int, int],
) -> np.ndarray:
    """Fast mask processing: crop in prototype space."""
    c, mh, mw = protos.shape
    ih, iw = input_shape
    h_orig, w_orig = original_shape

    masks = _sigmoid(masks_in @ protos.reshape(c, -1)).reshape(-1, mh, mw)
    scale_x, scale_y = mw / iw, mh / ih
    scaled_boxes = bboxes.copy()
    scaled_boxes[:, [0, 2]] *= scale_x
    scaled_boxes[:, [1, 3]] *= scale_y
    masks = crop_mask(masks, scaled_boxes)
    masks = (masks > 0.5).astype(np.uint8)

    gain = min(ih / h_orig, iw / w_orig)
    pad_x, pad_y = (iw - w_orig * gain) / 2, (ih - h_orig * gain) / 2
    top = int(round(pad_y * scale_y))
    left = int(round(pad_x * scale_x))
    bottom = int(round((ih - pad_y) * scale_y))
    right = int(round((iw - pad_x) * scale_x))
    masks = masks[:, top:bottom, left:right]

    result = np.empty((len(masks), h_orig, w_orig), dtype=np.uint8)
    for i in range(len(masks)):
        result[i] = cv2.resize(masks[i], (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    return result
