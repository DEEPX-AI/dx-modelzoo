from __future__ import annotations

from typing import Any, Tuple

import cv2
import numpy as np

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.evaluator import EVALUATOR_REGISTRY, EvaluatorBase
from dx_modelzoo.evaluator.super_resolution_evaluator import calculate_psnr, calculate_ssim
from dx_modelzoo.session import SessionBase


@EVALUATOR_REGISTRY.register("low_light_enhancement")
class LowLightEnhancementEvaluator(EvaluatorBase):
    """Low-Light Image Enhancement Evaluator (PSNR/SSIM)."""

    def __init__(self, session: SessionBase, dataset: DatasetBase, **kwargs) -> None:
        super().__init__(session, dataset, **kwargs)

    def init_metrics(self) -> dict:
        return {"total_psnr": 0.0, "total_ssim": 0.0, "total_samples": 0}

    def extract_inputs(self, batch_data: Tuple) -> np.ndarray:
        low_images, high_images = batch_data
        return low_images

    def process_batch_result(self, batch_data: Tuple, output: Any, metrics_state: dict) -> dict:
        low_images, high_images = batch_data
        pred = np.asarray(output)
        if pred.ndim == 4:
            pred = pred[0]
        if pred.ndim == 3 and pred.shape[0] in (1, 3):
            pred = pred.transpose(1, 2, 0)
        gt = np.asarray(high_images)

        if gt.max() > 1.0:
            gt = gt.astype(np.float32) / 255.0
        pred = np.clip(pred, 0.0, 1.0)
        if pred.shape[:2] != gt.shape[:2]:
            gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]))
        psnr = calculate_psnr(pred, gt)
        ssim = calculate_ssim(pred, gt)
        metrics_state["total_psnr"] += psnr
        metrics_state["total_ssim"] += ssim
        metrics_state["total_samples"] += 1
        return metrics_state

    def compute_final_metrics(self, metrics_state: dict) -> dict:
        total = metrics_state["total_samples"]
        avg_psnr = metrics_state["total_psnr"] / total if total > 0 else 0.0
        avg_ssim = metrics_state["total_ssim"] / total if total > 0 else 0.0
        avg_fps = total / self.total_inference_time if self.total_inference_time > 0 else 0.0
        return self._finalize(
            metric_names=["PSNR", "SSIM"],
            metric_values=[avg_psnr, avg_ssim],
            fps=avg_fps,
        )

    def format_progress_desc(self, metrics_state: dict, current_fps: float) -> str:
        total = metrics_state["total_samples"]
        if total == 0:
            return "LLIE | Initializing..."
        avg_psnr = metrics_state["total_psnr"] / total
        avg_ssim = metrics_state["total_ssim"] / total
        return f"LLIE | PSNR:{avg_psnr:.2f}dB SSIM:{avg_ssim:.4f} FPS:{current_fps:.1f}"
