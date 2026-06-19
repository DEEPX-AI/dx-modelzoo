from __future__ import annotations

import math
from typing import Tuple

import cv2
import numpy as np

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.evaluator import EVALUATOR_REGISTRY, EvaluatorBase
from dx_modelzoo.session import SessionBase


@EVALUATOR_REGISTRY.register("depth_estimation")
class DepthEstimationEvaluator(EvaluatorBase):
    """Depth Estimation Evaluator (RMSE)."""

    def __init__(self, session: SessionBase, dataset: DatasetBase, **kwargs) -> None:
        super().__init__(session, dataset, workers=12, **kwargs)
        self.use_median_scaling = False
        self.use_scale_shift = False
        self.max_depth = 0.0
        self.eigen_crop = False

    def init_metrics(self) -> dict:
        return {"rmse_sum": 0.0, "count": 0}

    def extract_inputs(self, batch_data: Tuple) -> np.ndarray:
        images, depth = batch_data
        return images

    def process_batch_result(self, batch_data: Tuple, output: np.ndarray, metrics_state: dict) -> dict:
        images, depth = batch_data
        output = np.asarray(output, dtype=np.float64).squeeze()  # [1,1,H,W] → [H,W]
        depth = np.asarray(depth, dtype=np.float64).squeeze()  # [1,H,W] → [H,W]

        # Upscale prediction to GT resolution when shapes differ
        if output.shape != depth.shape:
            output = cv2.resize(
                output.astype(np.float32),
                (depth.shape[1], depth.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            ).astype(np.float64)

        # Alignment mask: valid GT pixels with finite predictions
        align_mask = (depth > 0) & np.isfinite(output)
        if self.max_depth > 0:
            align_mask &= depth <= self.max_depth
        if self.eigen_crop:
            h, w = depth.shape
            crop = np.zeros((h, w), dtype=bool)
            crop[int(0.09375 * h) : int(0.98125 * h), int(0.0640625 * w) : int(0.9390625 * w)] = True
            align_mask &= crop

        if not np.any(align_mask):
            metrics_state["rmse_sum"] += 0.0
            metrics_state["count"] += 1
            return metrics_state

        if self.use_scale_shift:
            pred_v = output[align_mask].ravel()
            gt_v = depth[align_mask].ravel()
            A = np.vstack([pred_v, np.ones_like(pred_v)]).T
            coeffs = np.linalg.lstsq(A, gt_v, rcond=None)[0]
            output = output * coeffs[0] + coeffs[1]
        elif self.use_median_scaling:
            scale = np.median(depth[align_mask]) / (np.median(output[align_mask]) + 1e-10)
            output = output * scale

        if self.max_depth > 0:
            output = np.clip(output, 0, self.max_depth)

        # RMSE mask: valid GT and positive aligned prediction
        eval_mask = align_mask & (output > 0)
        if not np.any(eval_mask):
            metrics_state["rmse_sum"] += 0.0
            metrics_state["count"] += 1
            return metrics_state

        mse = float(np.mean((output[eval_mask] - depth[eval_mask]) ** 2))
        rmse = math.sqrt(mse)

        metrics_state["rmse_sum"] += rmse
        metrics_state["count"] += 1
        return metrics_state

    def compute_final_metrics(self, metrics_state: dict) -> dict:
        count = metrics_state["count"]
        avg_rmse = metrics_state["rmse_sum"] / count if count > 0 else 0.0
        avg_fps = count / self.total_inference_time if self.total_inference_time > 0 else 0.0
        return self._finalize(
            metric_names=["RMSE"],
            metric_values=[avg_rmse],
            fps=avg_fps,
        )

    def format_progress_desc(self, metrics_state: dict, current_fps: float) -> str:
        return f"Depth | Current_FPS:{current_fps:.1f}"
