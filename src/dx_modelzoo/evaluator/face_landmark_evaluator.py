from __future__ import annotations

from typing import Any

import numpy as np

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.evaluator import EVALUATOR_REGISTRY, EvaluatorBase
from dx_modelzoo.session import SessionBase

YAW_BINS = [(0, 30), (30, 60), (60, 90)]


@EVALUATOR_REGISTRY.register("face_landmark")
class FaceLandmarkEvaluator(EvaluatorBase):
    """Face Landmark Evaluator using NME (Normalized Mean Error)."""

    def __init__(self, session: SessionBase, dataset: DatasetBase, **kwargs) -> None:
        super().__init__(session, dataset, workers=12, **kwargs)

    def init_metrics(self) -> dict:
        return {
            "nme_per_bin": {f"{lo}_{hi}": [] for lo, hi in YAW_BINS},
            "all_nme": [],
            "count": 0,
        }

    def extract_inputs(self, batch_data: Any) -> np.ndarray:
        image, gt_landmarks, bbox_size, yaw_deg, idx = batch_data
        return image

    def process_batch_result(self, batch_data: Any, output: Any, metrics_state: dict) -> dict:
        image, gt_landmarks, bbox_size, yaw_deg, idx = batch_data
        pred_landmarks = np.asarray(output)
        gt_landmarks = np.asarray(gt_landmarks)
        if gt_landmarks.ndim == 3:
            gt_landmarks = gt_landmarks[0]
        bbox_val = float(bbox_size.item() if hasattr(bbox_size, "item") else bbox_size)
        yaw_val = float(yaw_deg.item() if hasattr(yaw_deg, "item") else yaw_deg)
        distances = np.sqrt(np.sum((pred_landmarks - gt_landmarks) ** 2, axis=1))
        nme = np.mean(distances) / max(bbox_val, 1e-6) * 100
        metrics_state["all_nme"].append(nme)
        metrics_state["count"] += 1
        for lo, hi in YAW_BINS:
            if lo <= yaw_val < hi or (hi == 90 and yaw_val >= 60):
                metrics_state["nme_per_bin"][f"{lo}_{hi}"].append(nme)
                break
        return metrics_state

    def compute_final_metrics(self, metrics_state: dict) -> dict:
        count = metrics_state["count"]
        avg_fps = count / self.total_inference_time if self.total_inference_time > 0 else 0.0
        mean_nme = float(np.mean(metrics_state["all_nme"])) if metrics_state["all_nme"] else 0.0
        return self._finalize(
            metric_names=["NME"],
            metric_values=[mean_nme],
            fps=avg_fps,
        )

    def format_progress_desc(self, metrics_state: dict, current_fps: float) -> str:
        count = metrics_state.get("count", 0)
        avg_nme = float(np.mean(metrics_state["all_nme"])) if metrics_state["all_nme"] else 0.0
        return f"FaceLandmark | NME:{avg_nme:.4f} Count:{count} FPS:{current_fps:.1f}"
