"""Custom ops for NVIDIA Isaac DOPE (HOPE-Image, 6-DoF pose).

Registers the model-local postprocessing op:

* ``dope_decode`` — decodes the single DOPE output tensor ``[1,25,H,W]`` (e.g.
  ``[1,25,60,80]`` for 480x640 input, stride 8) into one normalised cuboid
  detection (9 belief-map peaks + sub-pixel refinement) and, when camera
  intrinsics and the object cuboid are supplied via the postprocessing context,
  recovers the full 6-DoF pose via PnP — DOPE's official 2D->3D conversion
  (``cv2.solvePnP`` on the cuboid).

DOPE output tensor ``[1,25,H,W]``:
    channels 0..8   : 9 belief maps (8 cuboid corners + centroid; centroid = ch 8)
    channels 9..24  : 16 affinity-field channels (8 vertices x 2) -> centroid
The network is trained on exactly one object, so global-argmax peak-picking on
the 9 belief maps is sufficient (affinity fields only disambiguate multiple
instances, which single-class HOPE scenes do not contain).

Coordinates are normalised by image (W, H); the squash-resize used to feed the
network is therefore transparent to the metric. The recovered pose is expressed
in the DOPE cuboid frame and scored (ADD / ADD-S + 2D keypoint metrics) by the
shared ``object_pose_estimation`` evaluator (``single`` variant).
"""

from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np

from dx_modelzoo.postprocessing import POSTPROCESSING_REGISTRY

CENTROID_CHANNEL = 8


@POSTPROCESSING_REGISTRY.register("dope_decode")
class DopeDecode:
    """Decode DOPE belief/affinity maps -> normalised cuboid detection + pose.

    The decode step is always performed. When the postprocessing context
    supplies ``intrinsics`` (3x3 K), ``image_wh`` (W, H) and ``cuboid3d`` (9x3
    cm), the 6-DoF pose is additionally recovered via ``cv2.solvePnP`` (EPnP) —
    the official DOPE 2D->3D conversion — and returned under ``"pose"``.
    """

    def __init__(self, peak_thresh: float = 0, refine: bool = True, **kwargs) -> None:
        self.peak_thresh = peak_thresh
        self.refine = refine
        self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    @staticmethod
    def _subpixel(m: np.ndarray, iy: int, ix: int) -> tuple:
        """3x3 belief-weighted centroid refinement around (iy, ix)."""
        h, w = m.shape
        y0, y1 = max(iy - 1, 0), min(iy + 2, h)
        x0, x1 = max(ix - 1, 0), min(ix + 2, w)
        patch = np.clip(m[y0:y1, x0:x1], 0, None)
        s = float(patch.sum())
        if s <= 1e-09:
            return float(ix), float(iy)
        ys = np.arange(y0, y1)
        xs = np.arange(x0, x1)
        cy = float((patch.sum(axis=1) * ys).sum() / s)
        cx = float((patch.sum(axis=0) * xs).sum() / s)
        return cx, cy

    def _solve_pose(
        self,
        kps_norm: np.ndarray,
        cuboid3d: np.ndarray,
        K: np.ndarray,
        width: int,
        height: int,
    ) -> Optional[Dict[str, np.ndarray]]:
        """Recover (R, t) from the 9 normalised keypoints via EPnP.

        Returns ``{"R": (3,3), "t": (3,)}`` (DOPE cuboid frame) or ``None`` on
        failure / degenerate (behind-camera) solutions.
        """
        img_pts = kps_norm.astype(np.float64).copy()
        img_pts[:, 0] *= float(width)
        img_pts[:, 1] *= float(height)
        try:
            ok, rvec, tvec = cv2.solvePnP(
                np.asarray(cuboid3d, dtype=np.float64),
                img_pts,
                np.asarray(K, dtype=np.float64),
                self._dist_coeffs,
                flags=cv2.SOLVEPNP_EPNP,
            )
        except cv2.error:
            return None
        if not ok:
            return None
        t = tvec.reshape(3)
        if t[2] <= 0:  # behind camera — degenerate
            return None
        R, _ = cv2.Rodrigues(rvec)
        return {"R": R, "t": t}

    def __call__(self, outputs, **kwargs) -> Dict[str, np.ndarray]:
        arr = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        arr = np.asarray(arr)
        if arr.ndim == 4:
            arr = arr[0]
        belief = arr[:9]
        h, w = belief.shape[1], belief.shape[2]
        kps = np.zeros((9, 2), dtype=np.float64)
        confs = np.zeros(9, dtype=np.float64)
        for c in range(9):
            m = belief[c]
            iy, ix = np.unravel_index(int(np.argmax(m)), m.shape)
            confs[c] = float(m[iy, ix])
            if self.refine:
                cx, cy = self._subpixel(m, int(iy), int(ix))
            else:
                cx, cy = float(ix), float(iy)
            kps[c, 0] = (cx + 0.5) / w
            kps[c, 1] = (cy + 0.5) / h

        result: Dict[str, object] = {
            "keypoints": kps,
            "centroid": kps[CENTROID_CHANNEL].copy(),
            "confidence": float(confs[CENTROID_CHANNEL]),
            "all_conf": confs,
            "pose": None,
        }

        # Official DOPE 2D->3D conversion (PnP) when geometry/intrinsics given.
        intrinsics = kwargs.get("intrinsics")
        image_wh = kwargs.get("image_wh")
        cuboid3d = kwargs.get("cuboid3d")
        if intrinsics is not None and image_wh is not None and cuboid3d is not None:
            iw, ih = int(image_wh[0]), int(image_wh[1])
            result["pose"] = self._solve_pose(kps, cuboid3d, intrinsics, iw, ih)
        return result
