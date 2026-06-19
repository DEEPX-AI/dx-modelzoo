from __future__ import annotations

from typing import Any

import numpy as np

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.evaluator import EVALUATOR_REGISTRY, EvaluatorBase
from dx_modelzoo.session import SessionBase

WRIST = 0
MIDDLE_FINGER_MCP = 9


@EVALUATOR_REGISTRY.register("hand_landmark")
class HandLandmarkEvaluator(EvaluatorBase):
    """Hand Landmark Evaluator using MNAE."""

    def __init__(self, session: SessionBase, dataset: DatasetBase, **kwargs) -> None:
        super().__init__(session, dataset, workers=12, **kwargs)

    def init_metrics(self) -> dict:
        return {"total_mnae": 0.0, "count": 0}

    def extract_inputs(self, batch_data: Any) -> np.ndarray:
        image, gt_kpts, idx = batch_data
        return image

    def process_batch_result(self, batch_data: Any, output: Any, metrics_state: dict) -> dict:
        image, gt_kpts, idx = batch_data
        pred_kpts = output[0] if isinstance(output, (list, tuple)) else output
        pred_kpts = np.asarray(pred_kpts).squeeze()
        # Reshape flat output to [K, 3] keypoints
        if pred_kpts.ndim == 1:
            pred_kpts = pred_kpts.reshape(-1, 3)
        elif pred_kpts.ndim == 3:
            pred_kpts = pred_kpts[0]
        gt_kpts = np.asarray(gt_kpts)
        if gt_kpts.ndim == 3:
            gt_kpts = gt_kpts[0]
        pred_xy = pred_kpts[:, :2]
        # Model outputs pixel coordinates; normalize to [0,1] to match GT.
        image_arr = np.asarray(image)
        # Determine spatial width: handle both NCHW/CHW and HWC/NHWC layouts
        if image_arr.ndim == 4:
            input_size = image_arr.shape[-1] if image_arr.shape[1] <= 4 else image_arr.shape[-2]
        elif image_arr.ndim == 3:
            input_size = image_arr.shape[1] if image_arr.shape[0] <= 4 else image_arr.shape[-2]
        else:
            input_size = image_arr.shape[-1]
        pred_xy = pred_xy / input_size
        gt_xy = gt_kpts[:, :2]
        visibility = gt_kpts[:, 2]
        palm_size = np.sqrt(np.sum((gt_xy[WRIST] - gt_xy[MIDDLE_FINGER_MCP]) ** 2))
        visible_mask = visibility > 0
        if visible_mask.sum() > 0 and palm_size > 1e-6:
            distances = np.sqrt(np.sum((pred_xy[visible_mask] - gt_xy[visible_mask]) ** 2, axis=1))
            mnae = np.mean(distances) / palm_size
        else:
            mnae = 0.0
        metrics_state["total_mnae"] += mnae
        metrics_state["count"] += 1
        return metrics_state

    def compute_final_metrics(self, metrics_state: dict) -> dict:
        count = metrics_state["count"]
        avg_mnae = metrics_state["total_mnae"] / max(count, 1)
        avg_fps = count / self.total_inference_time if self.total_inference_time > 0 else 0.0
        return self._finalize(
            metric_names=["MNAE"],
            metric_values=[avg_mnae],
            fps=avg_fps,
        )

    def format_progress_desc(self, metrics_state: dict, current_fps: float) -> str:
        count = metrics_state.get("count", 0)
        avg_mnae = metrics_state["total_mnae"] / max(count, 1) if count > 0 else 0.0
        return f"HandLandmark | MNAE:{avg_mnae:.6f} Count:{count} FPS:{current_fps:.1f}"
