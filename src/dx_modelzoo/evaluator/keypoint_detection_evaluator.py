from __future__ import annotations

from typing import Any, Optional, Tuple

import cv2
import numpy as np

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.evaluator import EVALUATOR_REGISTRY, EvaluatorBase
from dx_modelzoo.session import SessionBase

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_CELL = 8  # SuperPoint / XFeat cell size
_NMS_DIST = 4
_CONF_THRESH = 0.015
_MAX_KP = 2048
_RANSAC_REPROJ = 3.0
_HEA_THRESH = 3.0  # corner reprojection error threshold (px)
_MSCORE_THRESH = 3.0  # correct-match distance threshold (px)

# Preprocessed image size used by both SuperPoint and XFeat YAMLs
_PREP_H, _PREP_W = 480, 640


# ---------------------------------------------------------------------------
# Keypoint decoding (65-channel dustbin heatmap → pixel keypoints)
# ---------------------------------------------------------------------------
def _decode_heatmap(semi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    nodust = soft[:-1].reshape(_CELL, _CELL, Hc, Wc)
    heatmap = nodust.transpose(2, 0, 3, 1).reshape(Hc * _CELL, Wc * _CELL)

    # NMS via dilation
    kernel = np.ones((2 * _NMS_DIST + 1, 2 * _NMS_DIST + 1), np.uint8)
    dilated = cv2.dilate(heatmap.astype(np.float32), kernel)
    heatmap = heatmap * (heatmap == dilated)

    # Threshold + top-K
    ys, xs = np.where(heatmap > _CONF_THRESH)
    scores = heatmap[ys, xs]
    if len(scores) > _MAX_KP:
        top = np.argpartition(scores, -_MAX_KP)[-_MAX_KP:]
        xs, ys, scores = xs[top], ys[top], scores[top]

    return np.stack([xs, ys], axis=1).astype(np.float32), scores


# ---------------------------------------------------------------------------
# Descriptor sampling (bilinear interpolation on 1/8-res feature map)
# ---------------------------------------------------------------------------
def _sample_descriptors(desc_map: np.ndarray, kpts: np.ndarray) -> np.ndarray:
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

    x = kpts[:, 0].astype(np.float64) / _CELL
    y = kpts[:, 1].astype(np.float64) / _CELL
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


# ---------------------------------------------------------------------------
# Feature extraction from raw model outputs
# ---------------------------------------------------------------------------
def _extract_features(outputs: list) -> Tuple[np.ndarray, np.ndarray]:
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

    kpts, _scores = _decode_heatmap(semi)
    if desc is not None and len(kpts) > 0:
        descs = _sample_descriptors(desc, kpts)
    else:
        descs = np.zeros((len(kpts), 1), np.float32)
    return kpts, descs


# ---------------------------------------------------------------------------
# Matching & metrics
# ---------------------------------------------------------------------------
def _match_mnn(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    """Mutual nearest-neighbour matching (cosine similarity).

    Returns ``[M, 2]`` index pairs.
    """
    if len(d1) == 0 or len(d2) == 0:
        return np.zeros((0, 2), dtype=int)
    sim = d1 @ d2.T
    nn12 = np.argmax(sim, axis=1)
    nn21 = np.argmax(sim, axis=0)
    ids = np.arange(len(d1))
    mask = nn21[nn12] == ids
    return np.stack([ids[mask], nn12[mask]], axis=1)


def _scale_homography(
    H_gt: np.ndarray,
    orig_hw: Tuple[int, int],
) -> np.ndarray:
    """Scale ground-truth homography from original to preprocessed coords."""
    sy = _PREP_H / orig_hw[0]
    sx = _PREP_W / orig_hw[1]
    S = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)
    return S @ H_gt @ np.linalg.inv(S)


def _compute_hea(
    kpts1: np.ndarray,
    kpts2: np.ndarray,
    matches: np.ndarray,
    H: np.ndarray,
) -> bool:
    """True if estimated homography reprojects corners within threshold."""
    if len(matches) < 4:
        return False
    src = kpts1[matches[:, 0]].astype(np.float64)
    dst = kpts2[matches[:, 1]].astype(np.float64)
    H_est, _ = cv2.findHomography(src, dst, cv2.RANSAC, _RANSAC_REPROJ)
    if H_est is None:
        return False
    corners = np.array(
        [[0, 0], [_PREP_W - 1, 0], [_PREP_W - 1, _PREP_H - 1], [0, _PREP_H - 1]],
        dtype=np.float64,
    ).reshape(1, -1, 2)
    try:
        c_gt = cv2.perspectiveTransform(corners, H.astype(np.float64))
        c_est = cv2.perspectiveTransform(corners, H_est)
    except cv2.error:
        return False
    err = np.mean(np.linalg.norm(c_gt.reshape(-1, 2) - c_est.reshape(-1, 2), axis=1))
    return bool(err < _HEA_THRESH)


def _compute_mscore(
    kpts1: np.ndarray,
    kpts2: np.ndarray,
    matches: np.ndarray,
    H: np.ndarray,
) -> float:
    """Matching score = correct matches / min(N1, N2)."""
    denom = min(len(kpts1), len(kpts2))
    if denom == 0 or len(matches) == 0:
        return 0.0
    src = kpts1[matches[:, 0]].astype(np.float64)
    dst = kpts2[matches[:, 1]].astype(np.float64)
    src_h = np.hstack([src, np.ones((len(src), 1))])
    warped = (H @ src_h.T).T
    w = warped[:, 2:3]
    w = np.where(np.abs(w) < 1e-8, 1e-8, w)
    warped_xy = warped[:, :2] / w
    errs = np.linalg.norm(warped_xy - dst, axis=1)
    return float((errs < _MSCORE_THRESH).sum()) / denom


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------
@EVALUATOR_REGISTRY.register("keypoint_detection")
class KeypointDetectionEvaluator(EvaluatorBase):
    """Local-feature evaluator — Homography Estimation Accuracy & Matching Score.

    For each HPatches sequence the reference image (n=1) is matched
    against the five target images (n=2..6) using mutual nearest
    neighbour descriptor matching followed by RANSAC homography
    estimation.

    Metrics:
        * **HEA** — fraction of pairs where estimated homography
          reprojects image corners within 3 px of the ground-truth H.
        * **M.Score** — mean ratio of correct matches (warp error < 3 px)
          to min(N_ref, N_tgt) across all pairs.
    """

    def __init__(self, session: SessionBase, dataset: DatasetBase, **kwargs) -> None:
        super().__init__(session, dataset, **kwargs)

    # -- state management ---------------------------------------------------

    def init_metrics(self) -> dict:
        return {
            "hea_correct": 0,
            "hea_total": 0,
            "mscore_sum": 0.0,
            "mscore_count": 0,
            "sample_count": 0,
            # per-sequence scratch (reset each sequence)
            "_ref_kpts": None,
            "_ref_descs": None,
            "_ref_shape": None,
            "_ref_seq": None,
        }

    # -- per-sample ---------------------------------------------------------

    def extract_inputs(self, batch_data: Tuple) -> np.ndarray:
        image = batch_data[0]
        if isinstance(image, np.ndarray) and image.ndim == 3:
            image = np.expand_dims(image, 0)
        return image

    def process_batch_result(
        self,
        batch_data: Tuple,
        output: Any,
        metrics_state: dict,
    ) -> dict:
        metrics_state["sample_count"] += 1
        seq = batch_data[2]
        n = int(batch_data[3])
        orig_shape = batch_data[1]
        H_1_n = batch_data[5] if len(batch_data) > 5 else None

        kpts, descs = output

        if n == 1:
            metrics_state["_ref_kpts"] = kpts
            metrics_state["_ref_descs"] = descs
            metrics_state["_ref_shape"] = orig_shape
            metrics_state["_ref_seq"] = seq
            return metrics_state

        ref_kpts = metrics_state.get("_ref_kpts")
        ref_descs = metrics_state.get("_ref_descs")
        ref_shape = metrics_state.get("_ref_shape")
        if ref_kpts is None or metrics_state.get("_ref_seq") != seq or H_1_n is None:
            return metrics_state

        H_scaled = _scale_homography(H_1_n, (ref_shape[0], ref_shape[1]))
        matches = _match_mnn(ref_descs, descs)

        hea = _compute_hea(ref_kpts, kpts, matches, H_scaled)
        mscore = _compute_mscore(ref_kpts, kpts, matches, H_scaled)

        metrics_state["hea_correct"] += int(hea)
        metrics_state["hea_total"] += 1
        metrics_state["mscore_sum"] += mscore
        metrics_state["mscore_count"] += 1

        return metrics_state

    # -- final metrics ------------------------------------------------------

    def compute_final_metrics(self, metrics_state: dict) -> dict:
        n_pairs = metrics_state["hea_total"]
        hea_pct = (metrics_state["hea_correct"] / n_pairs * 100) if n_pairs else 0.0
        mscore_pct = (metrics_state["mscore_sum"] / metrics_state["mscore_count"] * 100) if n_pairs else 0.0
        n = metrics_state["sample_count"]
        fps = n / self.total_inference_time if self.total_inference_time > 0 else 0
        return self._finalize(
            metric_names=["HEA", "M.Score"],
            metric_values=[hea_pct, mscore_pct],
            fps=fps,
        )

    def format_progress_desc(self, metrics_state: dict, current_fps: float) -> str:
        n_pairs = metrics_state["hea_total"]
        if n_pairs == 0:
            return "HPatches | Initializing..."
        hea = metrics_state["hea_correct"] / n_pairs * 100
        ms = metrics_state["mscore_sum"] / metrics_state["mscore_count"] * 100
        return f"HPatches | HEA:{hea:.1f}% M.Score:{ms:.1f}% FPS:{current_fps:.1f}"
