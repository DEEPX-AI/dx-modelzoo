"""Shared box decoding utilities for anchor/prior-based detectors.

Provides prior box generation for SSD-style (RetinaFace, ULFGFD) and
grid generation for SCRFD-style detectors. All pure numpy, no torch.
"""
from __future__ import annotations

import itertools
import math
from typing import List, Sequence, Tuple

import numpy as np


def generate_ssd_priors(
    image_size: Tuple[int, int],
    steps: Sequence[int] = (8, 16, 32),
    min_sizes: Sequence[Sequence[int]] = ((16, 32), (64, 128), (256, 512)),
) -> np.ndarray:
    """Generate SSD-style prior boxes (anchors) for RetinaFace.

    Args:
        image_size: (height, width) of the model input.
        steps: Stride for each feature map level.
        min_sizes: Anchor sizes for each feature map level.

    Returns:
        np.ndarray of shape [N, 4] in (cx, cy, w, h) normalized format.
    """
    h, w = image_size
    anchors = []
    for k, step in enumerate(steps):
        fh, fw = h // step, w // step
        for i, j in itertools.product(range(fh), range(fw)):
            for ms in min_sizes[k]:
                cx = (j + 0.5) * step / w
                cy = (i + 0.5) * step / h
                s_w = ms / w
                s_h = ms / h
                anchors.append([cx, cy, s_w, s_h])
    return np.array(anchors, dtype=np.float32)


# ULFGFD uses different anchor config than RetinaFace
ULFGFD_STEPS = [8, 16, 32, 64]
ULFGFD_MIN_SIZES = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]


def generate_ulfgfd_priors(
    image_size: Tuple[int, int],
    steps: Sequence[int] = ULFGFD_STEPS,
    min_sizes: Sequence[Sequence[int]] = ULFGFD_MIN_SIZES,
) -> np.ndarray:
    """Generate ULFGFD-specific SSD prior boxes.

    Uses ceil division for feature map sizes (unlike RetinaFace's floor division).

    Returns:
        np.ndarray of shape [N, 4] in (cx, cy, w, h) normalized format.
    """
    h, w = image_size
    priors = []
    for k, step in enumerate(steps):
        fh = math.ceil(h / step)
        fw = math.ceil(w / step)
        for i, j in itertools.product(range(fh), range(fw)):
            for ms in min_sizes[k]:
                cx = (j + 0.5) * step / w
                cy = (i + 0.5) * step / h
                priors.append([cx, cy, ms / w, ms / h])
    return np.array(priors, dtype=np.float32)


def decode_ssd_boxes(
    loc: np.ndarray,
    priors: np.ndarray,
    variances: Tuple[float, float] = (0.1, 0.2),
) -> np.ndarray:
    """Decode SSD-encoded locations using prior boxes.

    Args:
        loc: Predicted locations [N, 4] (offsets from priors).
        priors: Prior boxes [N, 4] in (cx, cy, w, h) normalized format.
        variances: Variance values for center and size decoding.

    Returns:
        np.ndarray [N, 4] in (x1, y1, x2, y2) format.
    """
    boxes = np.concatenate(
        [
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * np.exp(np.clip(loc[:, 2:] * variances[1], -50.0, 50.0)),
        ],
        axis=1,
    )
    # Convert from (cx, cy, w, h) to (x1, y1, x2, y2)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def generate_scrfd_grids(
    h: int,
    w: int,
    strides: Sequence[int] = (8, 16, 32),
    num_anchors: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate SCRFD grid coordinates and stride arrays.

    SCRFD uses grid-based anchor-free detection with repeated anchors per cell.

    Args:
        h: Model input height.
        w: Model input width.
        strides: Stride for each feature map level.
        num_anchors: Number of anchors per grid cell.

    Returns:
        Tuple of (grids [N, 2], strides [N, 1]) where N = total grid cells * num_anchors.
    """
    all_grids: List[np.ndarray] = []
    all_strides: List[np.ndarray] = []
    for stride in strides:
        fh, fw = h // stride, w // stride
        if fh <= 0 or fw <= 0:
            continue
        gy, gx = np.meshgrid(np.arange(fh), np.arange(fw), indexing="ij")
        grid = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)
        # Repeat each grid cell for num_anchors
        grid = np.repeat(grid, num_anchors, axis=0)
        stride_arr = np.full((grid.shape[0], 1), stride, dtype=np.float32)
        all_grids.append(grid)
        all_strides.append(stride_arr)
    if not all_grids:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 1), dtype=np.float32)
    return np.concatenate(all_grids, axis=0), np.concatenate(all_strides, axis=0)
