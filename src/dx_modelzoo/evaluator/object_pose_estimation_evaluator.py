"""Unified object 6-DoF pose evaluator (``object_pose_estimation``).

A single evaluator that covers two output regimes, selected by the YAML
``evaluator.options.type`` field:

  * ``single`` — single-object belief-map models (NVIDIA Isaac DOPE on
    HOPE-Image). The postprocessed output is the ``dope_decode`` detection dict
    (``keypoints (9,2)``, ``centroid``, ``confidence`` and an optional recovered
    ``pose``); GT is the HOPE dict. The detection is matched to the nearest GT
    instance by projected-centroid distance, then scored **capability-based**
    with one metric per regime:

      - **3D**: ``ADD`` / ``ADD-S`` (cm) vs the GT pose — only when the
        prediction carries a recovered ``pose`` and the dataset exposes the
        object cuboid.
      - **2D**: ``kp_acc@0.1`` — PCK pass rate (mean normalised keypoint error
        below the threshold), GT-aware — only when both predicted and projected
        GT keypoints exist.

  * ``multi`` — multi-instance cuboid models (CenterPose on Objectron). Output
    is the 3-head ``(bboxes, scores, kps)`` tensor set; GT is ``[N, 9, 2]``
    keypoints (centroid + 8 corners). Detections are greedily matched to GT by
    centroid distance and the 8 cuboid corners are scored. Metrics:
    ``MPE_2D_pct``, ``Acc@0.2``, ``Acc@0.1``, ``DetRate``.

The 3D pose recovery (DOPE's official 2D->3D PnP conversion) lives in the
model's ``dope_decode`` postprocessing op; this evaluator stays a generic
scorer that compares whatever the model produced against the GT.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from dx_modelzoo.evaluator import EVALUATOR_REGISTRY, EvaluatorBase


@EVALUATOR_REGISTRY.register("object_pose_estimation")
class ObjectPoseEstimationEvaluator(EvaluatorBase):
    """Object 6-DoF pose evaluator (single-object DOPE / multi-instance CenterPose).

    Variant is chosen by ``evaluator.options.type`` (``single`` | ``multi``).
    """

    # Variant selector (set from evaluator.options.type).
    type: str = "single"

    # --- single (DOPE) config -------------------------------------------------
    score_thres: float = 0.0
    match_gate: float = 0.25  # max normalised 2D-centroid distance for a match
    kp_acc_thresh: float = 0.1  # PCK threshold on the normalised mean keypoint error

    # --- multi (CenterPose) config -------------------------------------------
    kp_grid: float = 128.0  # 512 input / stride 4
    score_floor: float = 0.1  # ignore detections below this confidence
    center_thresh: float = 0.2  # max normalised centroid distance for a valid match
    acc_thresh: float = 0.1  # mean corner error below this = correct (Objectron-style)
    acc_thresh2: float = 0.2  # matches the ONNX reference acc@0.2 threshold

    def __init__(self, session, dataset, **kwargs) -> None:
        super().__init__(session, dataset, **kwargs)

    @property
    def _is_multi(self) -> bool:
        return str(self.type).lower() in ("multi", "centerpose", "objectron")

    # ------------------------------------------------------------------ shared
    def init_metrics(self) -> dict:
        if self._is_multi:
            return {
                "mpe_sum": 0.0,
                "matched": 0,
                "correct": 0,
                "correct2": 0,
                "total_gt": 0,
            }
        return {
            "gt_total": 0,
            "pose_valid": 0,  # matched AND PnP succeeded
            "add_sum": 0.0,  # 3D ADD / ADD-S accumulator (cm)
            "kp_correct": 0,  # matched AND mean keypoint error < kp_acc_thresh
        }

    def extract_inputs(self, batch_data: Tuple) -> np.ndarray:
        img = batch_data[0]
        return np.ascontiguousarray(img) if self._is_multi else img

    def _build_postprocessing_context(self, batch_data) -> dict:
        """Forward camera intrinsics + object cuboid so ``dope_decode`` can PnP.

        Only the ``single`` (DOPE) regime needs this; the ``multi`` regime takes
        no extra context. ``cuboid3d`` is read from the dataset (``None`` when the
        object's published dimensions are unknown), in which case the
        postprocessor skips PnP and emits a ``pose=None`` detection.
        """
        if self._is_multi:
            return {}
        gt = batch_data[1]
        cuboid3d = getattr(self.dataset, "cuboid3d", None)
        if cuboid3d is None or not isinstance(gt, dict):
            return {}
        return {
            "intrinsics": np.asarray(gt["K"], dtype=np.float64),
            "image_wh": (int(gt["W"]), int(gt["H"])),
            "cuboid3d": cuboid3d,
        }

    def process_batch_result(
        self, batch_data: Tuple, output: Any, metrics_state: dict
    ) -> dict:
        if self._is_multi:
            return self._process_multi(batch_data, output, metrics_state)
        return self._process_single(batch_data, output, metrics_state)

    def compute_final_metrics(self, metrics_state: dict) -> dict:
        if self._is_multi:
            return self._final_multi(metrics_state)
        return self._final_single(metrics_state)

    def format_progress_desc(self, metrics_state: dict, current_fps: float) -> str:
        if self._is_multi:
            matched = metrics_state["matched"]
            total = metrics_state["total_gt"]
            mpe = (metrics_state["mpe_sum"] / matched) if matched else 0.0
            det = (matched / total * 100.0) if total else 0.0
            return (
                f"MPE={mpe:.4f} Det={det:.0f}% ({matched}/{total}) {current_fps:.1f}fps"
            )
        gt = max(metrics_state["gt_total"], 1)
        pose_valid = metrics_state["pose_valid"]
        add = metrics_state["add_sum"] / pose_valid if pose_valid else 0.0
        kp_acc = metrics_state["kp_correct"] / gt * 100
        return (
            f"DOPE/HOPE | ADD:{add:.2f}cm kp_acc@0.1:{kp_acc:.0f}% "
            f"FPS:{current_fps:.1f}"
        )

    # ------------------------------------------------------- single (DOPE)
    @staticmethod
    def _to_detection(output: Any) -> Dict[str, Any]:
        """Normalize the postprocessed output into a single detection dict.

        ``dope_decode`` returns ``{"keypoints": (9,2), "centroid": (2,),
        "confidence": float, "all_conf": (9,), "pose": {...}|None}`` in
        normalised [0,1] coords. Falls back to decoding a raw ``[1,25,H,W]``
        array (keypoints only, no pose) if no postprocessing was applied.
        """
        if isinstance(output, dict):
            return output
        arr = np.asarray(output[0] if isinstance(output, (list, tuple)) else output)
        if arr.ndim == 4:
            arr = arr[0]
        belief = arr[:9]
        kps: List = []
        confs: List = []
        for c in range(9):
            m = belief[c]
            iy, ix = np.unravel_index(int(np.argmax(m)), m.shape)
            kps.append(((ix + 0.5) / m.shape[1], (iy + 0.5) / m.shape[0]))
            confs.append(float(m[iy, ix]))
        kps = np.array(kps, dtype=np.float64)
        return {
            "keypoints": kps,
            "centroid": kps[8],
            "confidence": confs[8],
            "all_conf": np.array(confs),
            "pose": None,
        }

    def _add(self, R_p, t_p, R_g, t_g, corners3d, is_symmetric) -> float:
        """ADD (asymmetric) or ADD-S (symmetric) over the 8 cuboid corners (cm)."""
        pred = (corners3d @ R_p.T) + t_p  # (8, 3)
        gt = (corners3d @ R_g.T) + t_g  # (8, 3)
        if is_symmetric:
            d = np.linalg.norm(pred[:, None, :] - gt[None, :, :], axis=2)  # (8, 8)
            return float(d.min(axis=1).mean())
        return float(np.linalg.norm(pred - gt, axis=1).mean())

    def _process_single(
        self, batch_data: Tuple, output: Any, metrics_state: dict
    ) -> dict:
        _image, gt, _meta = batch_data
        n = int(gt["num"])
        metrics_state["gt_total"] += n
        if n == 0:
            return metrics_state

        det = self._to_detection(output)
        det_centroid = np.asarray(det["centroid"], dtype=np.float64)
        det_conf = float(det.get("confidence", 0.0))
        if det_conf < self.score_thres:
            return metrics_state

        # Match the single detection to the nearest GT instance (one match only).
        centroids = np.asarray(gt["centroids_2d"], dtype=np.float64)
        dists = np.linalg.norm(centroids - det_centroid, axis=1)
        gi = int(np.argmin(dists))
        if dists[gi] > self.match_gate:
            return metrics_state

        # --- 3D pose metric (ADD / ADD-S) — only if a pose was recovered -----
        cuboid3d = getattr(self.dataset, "cuboid3d", None)
        pose = det.get("pose")
        if pose is not None and cuboid3d is not None:
            frame_align = getattr(self.dataset, "frame_align", np.eye(3))
            is_symmetric = bool(getattr(self.dataset, "is_symmetric", False))
            corners3d = np.asarray(cuboid3d, dtype=np.float64)[:8]
            R_p = np.asarray(pose["R"], dtype=np.float64)
            t_p = np.asarray(pose["t"], dtype=np.float64).reshape(3)
            gt_pose = np.asarray(gt["poses"][gi], dtype=np.float64)
            R_g = gt_pose[:3, :3] @ frame_align
            t_g = gt_pose[:3, 3]
            add = self._add(R_p, t_p, R_g, t_g, corners3d, is_symmetric)
            metrics_state["pose_valid"] += 1
            metrics_state["add_sum"] += add

        # --- 2D keypoint metric — only if both sides have keypoints ----------
        gt_kps = np.asarray(
            gt.get("keypoints_2d", np.zeros((0, 9, 2))), dtype=np.float64
        )
        det_kps = det.get("keypoints")
        if det_kps is not None and gt_kps.ndim == 3 and gt_kps.shape[0] > gi:
            pred_kps = np.asarray(det_kps, dtype=np.float64)
            mpe = float(np.linalg.norm(pred_kps - gt_kps[gi], axis=1).mean())
            if mpe < self.kp_acc_thresh:
                metrics_state["kp_correct"] += 1
        return metrics_state

    def _final_single(self, metrics_state: dict) -> dict:
        gt_total = max(metrics_state["gt_total"], 1)
        pose_valid = metrics_state["pose_valid"]

        # 3D: ADD / ADD-S over recovered poses (cm).
        mean_add = (metrics_state["add_sum"] / pose_valid) if pose_valid else 0.0
        # 2D: keypoint accuracy (PCK), GT-aware — misses / failed matches count.
        kp_acc = metrics_state["kp_correct"] / gt_total * 100.0

        metric_name = "ADD-S" if getattr(self.dataset, "is_symmetric", False) else "ADD"
        avg_fps = (
            gt_total / self.total_inference_time if self.total_inference_time > 0 else 0
        )
        return self._finalize(
            metric_names=[metric_name, f"kp_acc@{self.kp_acc_thresh:g}"],
            metric_values=[mean_add, kp_acc],
            fps=avg_fps,
        )

    # ------------------------------------------------------- multi (CenterPose)
    @staticmethod
    def _as_list(output) -> List[np.ndarray]:
        if isinstance(output, (list, tuple)):
            return [np.asarray(o) for o in output]
        return [np.asarray(output)]

    def _process_multi(
        self, batch_data: Tuple, output: Any, metrics_state: dict
    ) -> dict:
        outs = self._as_list(output)
        # Locate heads by index (ONNX & DXNN share output order); fall back to shape.
        bboxes = outs[0]
        scores = outs[1]
        kps = outs[2]
        # Normalize to [N, ...]
        scores = scores.reshape(-1)
        n = scores.shape[0]
        kps = kps.reshape(n, -1)[:, :16].reshape(n, 8, 2) / self.kp_grid
        bb = bboxes.reshape(n, 4)
        centers = (bb[:, 0:2] + bb[:, 2:4]) / 2.0 / self.kp_grid

        gt_kpts = np.asarray(batch_data[1], dtype=np.float32)  # [num_inst, 9, 2]
        if gt_kpts.ndim != 3 or gt_kpts.shape[0] == 0:
            return metrics_state

        order = np.argsort(-scores)
        used: set = set()
        for g in range(gt_kpts.shape[0]):
            gt_center = gt_kpts[g, 0]
            gt_corners = gt_kpts[g, 1:9]
            metrics_state["total_gt"] += 1

            best = -1
            best_d = 1e9
            for pidx in order:
                if scores[pidx] < self.score_floor:
                    break  # sorted desc — rest are lower
                if pidx in used:
                    continue
                d = float(np.linalg.norm(centers[pidx] - gt_center))
                if d < best_d:
                    best_d = d
                    best = int(pidx)
            if best < 0 or best_d > self.center_thresh:
                continue  # no detection -> counts as miss for DetRate/Acc

            used.add(best)
            err = float(np.linalg.norm(kps[best] - gt_corners, axis=1).mean())
            metrics_state["mpe_sum"] += err
            metrics_state["matched"] += 1
            if err < self.acc_thresh:
                metrics_state["correct"] += 1
            if err < self.acc_thresh2:
                metrics_state["correct2"] += 1
        return metrics_state

    def _final_multi(self, metrics_state: dict) -> dict:
        matched = metrics_state["matched"]
        total = metrics_state["total_gt"]
        mpe = (metrics_state["mpe_sum"] / matched) if matched else 0.0
        acc = (metrics_state["correct"] / total * 100.0) if total else 0.0
        acc2 = (metrics_state["correct2"] / total * 100.0) if total else 0.0
        det = (matched / total * 100.0) if total else 0.0
        fps = (
            total / self.total_inference_time if self.total_inference_time > 0 else 0.0
        )
        # MPE reported as normalised and as % of image extent.
        return self._finalize(
            metric_names=["MPE_2D_pct", "Acc@0.2", "Acc@0.1", "DetRate"],
            metric_values=[mpe * 100.0, acc2, acc, det],
            fps=fps,
        )
