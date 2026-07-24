"""SFA3D (Super Fast and Accurate 3D Object Detection) output decoding.

Model outputs (NCHW, stride-4 BEV feature maps, 152x152):
  - hm_cen    (1, 3, 152, 152): per-class center heatmap
  - cen_offset(1, 2, 152, 152): sub-pixel center offset (dx, dy)
  - direction (1, 2, 152, 152): yaw angle as (cos, sin)
  - z_coor    (1, 1, 152, 152): z coordinate (height)
  - dim       (1, 3, 152, 152): 3D dimensions (h, w, l)

Decode: sigmoid heatmap → 3x3 max-pool peak NMS → top-k peaks →
BEV (cx, cy) + 3D (z, h, w, l, yaw) per detection.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter

from dx_modelzoo.postprocessing import POSTPROCESSING_REGISTRY
from dx_modelzoo.postprocessing.decode_utils import build_nms_input, sigmoid


@POSTPROCESSING_REGISTRY.register("sfa3d_decode")
class SFA3DDecode:
    """Decode SFA3D BEV heatmap outputs into 3D detections."""

    def __init__(
        self,
        input_size: int = 608,
        num_classes: int = 3,
        topk: int = 100,
        conf_thres: float = 0.5,
        **kwargs,
    ):
        self.input_size = input_size
        self.num_classes = num_classes
        self.topk = topk
        self.conf_thres = conf_thres

    def __call__(self, outputs, **kwargs):
        hm_cen, cen_offset, direction, z_coor, dim = outputs[:5]

        # Remove batch dim: (C, H, W)
        hm_cen = hm_cen[0]
        cen_offset = cen_offset[0]
        direction = direction[0]
        z_coor = z_coor[0]
        dim = dim[0]

        C, H, W = hm_cen.shape
        stride = self.input_size / H

        # cen_offset is a sigmoid-bounded sub-pixel offset in [0, 1] (SFA3D).
        cen_offset = sigmoid(cen_offset)

        # Sigmoid + peak NMS per class
        hm = sigmoid(hm_cen)
        # maximum_filter expects (H, W) per channel
        for c in range(C):
            pooled = maximum_filter(hm[c], size=3, mode="constant")
            hm[c] = hm[c] * (hm[c] == pooled).astype(hm.dtype)

        # Top-k across all classes
        flat = hm.ravel()
        k = min(self.topk, flat.size)
        top_idx = np.argpartition(-flat, k - 1)[:k]
        top_scores = flat[top_idx]

        mask = top_scores > self.conf_thres
        top_idx = top_idx[mask]
        top_scores = top_scores[mask]

        if top_idx.size == 0:
            return self._empty()

        cls_ids = top_idx // (H * W)
        spatial = top_idx % (H * W)
        ys = spatial // W
        xs = spatial % W

        # BEV center in pixel coords
        cx = (xs.astype(np.float64) + cen_offset[0, ys, xs]) * stride
        cy = (ys.astype(np.float64) + cen_offset[1, ys, xs]) * stride

        # 3D attributes
        z = z_coor[0, ys, xs].astype(np.float64)
        h3d = dim[0, ys, xs].astype(np.float64)
        w3d = dim[1, ys, xs].astype(np.float64)
        l3d = dim[2, ys, xs].astype(np.float64)

        # Yaw from (im, re) = (sin, cos): yaw = atan2(direction[0], direction[1])
        yaw = np.arctan2(direction[0, ys, xs], direction[1, ys, xs]).astype(np.float64)

        # ponytail: BEV pseudo-boxes for NMS compatibility; real 3D NMS if accuracy matters
        half_w = w3d / 2
        half_l = l3d / 2
        boxes = np.stack([cx - half_l, cy - half_w, cx + half_l, cy + half_w], axis=1)

        extra = np.stack([z, h3d, w3d, l3d, yaw], axis=1)
        return build_nms_input(
            boxes.astype(np.float64),
            top_scores.astype(np.float64),
            cls_ids.astype(np.float64),
            extra=extra,
        )

    def _empty(self):
        return build_nms_input(
            np.empty((0, 4), dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            extra=np.empty((0, 5), dtype=np.float64),
        )
