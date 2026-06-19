"""Shared model output decoding utilities.

Each model's custom_ops.py imports from here to build its decode pipeline.
This module itself is NOT registered as a postprocessing type.
"""
from __future__ import annotations

from typing import List

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))


def cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """[N, 4] center-x, center-y, width, height → x1, y1, x2, y2."""
    converted = boxes.copy()
    converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return converted


def xyxy_to_cxcywh(boxes: np.ndarray) -> np.ndarray:
    """[N, 4] x1, y1, x2, y2 → center-x, center-y, width, height."""
    converted = boxes.copy()
    converted[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2
    converted[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2
    converted[:, 2] = boxes[:, 2] - boxes[:, 0]
    converted[:, 3] = boxes[:, 3] - boxes[:, 1]
    return converted


def apply_obj_cls_score(output: np.ndarray) -> tuple:
    """YOLOv3/v5/v7 style: obj_conf × cls_scores.

    Input: [N, 5+C] (cx, cy, w, h, obj_conf, cls_0, cls_1, ...)
    Output: (boxes_xyxy [N,4], scores [N], class_ids [N])
    """
    boxes = cxcywh_to_xyxy(output[:, :4])
    obj_conf = output[:, 4]
    cls_scores = output[:, 5:] * obj_conf[:, None]
    scores = cls_scores.max(axis=1)
    class_ids = cls_scores.argmax(axis=1).astype(np.float64)
    return boxes, scores, class_ids


def split_box_cls(output: np.ndarray) -> tuple:
    """YOLOv8/v9/v11 style: [N, 4+C] (cx, cy, w, h, cls_0, cls_1, ...).

    No obj_conf - class scores are direct.
    Output: (boxes_xyxy [N,4], scores [N], class_ids [N])
    """
    boxes = cxcywh_to_xyxy(output[:, :4])
    cls_scores = output[:, 4:]
    scores = cls_scores.max(axis=1)
    class_ids = cls_scores.argmax(axis=1).astype(np.float64)
    return boxes, scores, class_ids


def split_box_cls_xyxy(output: np.ndarray) -> tuple:
    """Like split_box_cls but boxes are already in xyxy format.

    Input: [N, 4+C] (x1, y1, x2, y2, cls_0, cls_1, ...)
    Output: (boxes_xyxy [N,4], scores [N], class_ids [N])
    """
    boxes = output[:, :4].copy()
    cls_scores = output[:, 4:]
    scores = cls_scores.max(axis=1)
    class_ids = cls_scores.argmax(axis=1).astype(np.float64)
    return boxes, scores, class_ids


def transpose_output(output: np.ndarray) -> np.ndarray:
    """[1, C, N] → [1, N, C] transpose (when C < N)."""
    if output.ndim == 3 and output.shape[1] < output.shape[2]:
        return np.transpose(output, (0, 2, 1))
    return output


def build_nms_input(
    boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray, extra: np.ndarray | None = None
) -> dict:
    """Pack into NMS-expected dict format.

    Args:
        extra: Optional [N, K] array of extra per-detection data to
            carry through NMS (e.g. rotated box params).
    """
    d: dict = {"boxes": boxes, "scores": scores, "class_ids": class_ids}
    if extra is not None:
        d["extra"] = extra
    return d


# --- Grid/Anchor utilities (for YOLOX, YOLOv7, NanoDet, etc.) ---


def build_yolox_grids(input_size: int, strides: List[int]):
    """Pre-compute YOLOX grid offsets and stride arrays.
    Returns: (grids [1, N, 2], strides [1, N, 1])
    """
    grids, stride_arr = [], []
    for s in strides:
        sz = input_size // s
        yv, xv = np.meshgrid(np.arange(sz), np.arange(sz), indexing="ij")
        grid = np.stack((xv, yv), axis=2).reshape(1, -1, 2).astype(np.float32)
        grids.append(grid)
        stride_arr.append(np.full((1, grid.shape[1], 1), s, dtype=np.float32))
    return np.concatenate(grids, axis=1), np.concatenate(stride_arr, axis=1)


def generate_grid_center_priors(input_height: int, input_width: int, strides: List[int]) -> np.ndarray:
    """Generate center priors as (x, y, stride) for each grid cell."""
    center_priors = []
    for stride in strides:
        feat_w = int(np.ceil(input_width / stride))
        feat_h = int(np.ceil(input_height / stride))
        for y in range(feat_h):
            for x in range(feat_w):
                center_priors.append([x, y, stride])
    return np.array(center_priors, dtype=np.float32)


def infer_input_size(num_points: int, strides: List[int]) -> int:
    """Infer spatial input size from grid point count and strides.
    Solves: num_points = sum((size/stride)^2 for stride in strides)
    """
    inv_stride_sq_sum = sum(1.0 / (s * s) for s in strides)
    size = int(round((num_points / inv_stride_sq_sum) ** 0.5))
    return size


# YOLOv7 anchor configurations
YOLOV7_ANCHORS = {
    "yolov7": [
        [[12, 16], [19, 36], [40, 28]],
        [[36, 75], [76, 55], [72, 146]],
        [[142, 110], [192, 243], [459, 401]],
    ],
    "yolov7_w6": [
        [[19, 27], [44, 40], [38, 94]],
        [[96, 68], [86, 152], [180, 137]],
        [[140, 301], [303, 264], [238, 542]],
        [[436, 615], [739, 380], [925, 792]],
    ],
}
YOLOV7_STRIDES = [8, 16, 32, 64]
