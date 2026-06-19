"""SuperPoint / XFeat keypoint decoder."""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from dx_modelzoo.postprocessing import POSTPROCESSING_REGISTRY


class SuperPointDecode:
    """Decode 65-channel dustbin heatmap + descriptor map → keypoints + descriptors.

    Input: list of model output tensors (finds 65-ch heatmap and descriptor automatically)
    Output: (keypoints [N,2], descriptors [N,D])
    """

    _CELL = 8  # SuperPoint / XFeat cell size
    _NMS_DIST = 4
    _CONF_THRESH = 0.015
    _MAX_KP = 2048

    def __init__(
        self,
        cell: int = 8,
        conf_thresh: float = 0.015,
        max_kp: int = 2048,
        nms_dist: int = 4,
        **kwargs,
    ):
        self.cell = cell
        self.conf_thresh = conf_thresh
        self.max_kp = max_kp
        self.nms_dist = nms_dist

    def __call__(self, outputs, **kwargs):
        raw = outputs if isinstance(outputs, (list, tuple)) else [outputs]
        kpts, descs = self._extract_features(raw)
        return kpts, descs

    def _decode_heatmap(self, semi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Decode a 65-channel heatmap into pixel keypoints + scores.

        Args:
            semi: ``[1, 65, Hc, Wc]`` or ``[65, Hc, Wc]``.

        Returns:
            kpts ``[N, 2]`` (x, y) in pixel coords, scores ``[N]``.
        """
        if semi.ndim == 4:
            semi = semi[0]
        Hc, Wc = semi.shape[1], semi.shape[2]

        # Softmax over 65 channels (numerically stable)
        shifted = semi - semi.max(axis=0, keepdims=True)
        exp = np.exp(shifted)
        soft = exp / exp.sum(axis=0, keepdims=True)

        # Remove dustbin, reshape to pixel grid
        nodust = soft[:-1].reshape(self.cell, self.cell, Hc, Wc)
        heatmap = nodust.transpose(2, 0, 3, 1).reshape(Hc * self.cell, Wc * self.cell)

        # NMS via dilation
        kernel = np.ones((2 * self.nms_dist + 1, 2 * self.nms_dist + 1), np.uint8)
        dilated = cv2.dilate(heatmap.astype(np.float32), kernel)
        heatmap = heatmap * (heatmap == dilated)

        # Threshold + top-K
        ys, xs = np.where(heatmap > self.conf_thresh)
        scores = heatmap[ys, xs]
        if len(scores) > self.max_kp:
            top = np.argpartition(scores, -self.max_kp)[-self.max_kp :]
            xs, ys, scores = xs[top], ys[top], scores[top]

        return np.stack([xs, ys], axis=1).astype(np.float32), scores

    def _sample_descriptors(self, desc_map: np.ndarray, kpts: np.ndarray) -> np.ndarray:
        """Bilinear-interpolate descriptors at keypoint locations.

        Args:
            desc_map: ``[1, D, Hc, Wc]`` or ``[D, Hc, Wc]``.
            kpts: ``[N, 2]`` (x, y) in full-res pixel coords.

        Returns:
            ``[N, D]`` L2-normalised descriptors.
        """
        if desc_map.ndim == 4:
            desc_map = desc_map[0]
        D, Hc, Wc = desc_map.shape
        if len(kpts) == 0:
            return np.zeros((0, D), dtype=np.float32)

        x = kpts[:, 0].astype(np.float64) / self.cell
        y = kpts[:, 1].astype(np.float64) / self.cell
        x0 = np.floor(x).astype(int).clip(0, Wc - 1)
        y0 = np.floor(y).astype(int).clip(0, Hc - 1)
        x1 = (x0 + 1).clip(0, Wc - 1)
        y1 = (y0 + 1).clip(0, Hc - 1)

        dx = (x - x0).reshape(-1, 1)
        dy = (y - y0).reshape(-1, 1)

        d = (
            (1 - dx) * (1 - dy) * desc_map[:, y0, x0].T
            + dx * (1 - dy) * desc_map[:, y0, x1].T
            + (1 - dx) * dy * desc_map[:, y1, x0].T
            + dx * dy * desc_map[:, y1, x1].T
        )
        norms = np.linalg.norm(d, axis=1, keepdims=True)
        return (d / np.maximum(norms, 1e-8)).astype(np.float32)

    def _extract_features(self, outputs: list) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(kpts [N,2], descs [N,D])`` from raw model outputs.

        Handles both SuperPoint (semi, desc) and XFeat (feats, keypoints,
        heatmap) by locating the 65-channel tensor and the descriptor tensor.
        """
        semi: Optional[np.ndarray] = None
        desc: Optional[np.ndarray] = None

        for arr in outputs:
            a = np.asarray(arr)
            if a.ndim < 3:
                continue
            c = a.shape[1] if a.ndim == 4 else a.shape[0]
            if c == 65:
                semi = a
            elif c > 1:
                # Prefer the higher-dimensional descriptor (256 > 64)
                if desc is None:
                    desc = a
                else:
                    prev_c = desc.shape[1] if desc.ndim == 4 else desc.shape[0]
                    if c > prev_c:
                        desc = a

        if semi is None:
            return np.zeros((0, 2), np.float32), np.zeros((0, 1), np.float32)

        kpts, _scores = self._decode_heatmap(semi)
        if desc is not None and len(kpts) > 0:
            descs = self._sample_descriptors(desc, kpts)
        else:
            descs = np.zeros((len(kpts), 1), np.float32)
        return kpts, descs


if "superpoint_decode" not in POSTPROCESSING_REGISTRY:
    POSTPROCESSING_REGISTRY.register("superpoint_decode")(SuperPointDecode)
