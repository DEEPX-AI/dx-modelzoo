from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.evaluator import EVALUATOR_REGISTRY, EvaluatorBase
from dx_modelzoo.session import SessionBase

# Metric definitions follow PaddleSeg `Matting/ppmatting/metrics/metric.py`.
# Trimap-free models (PP-Matting) evaluate with `get_trimap: False`, which makes
# the whole image the unknown region, so no trimap masking is applied here.
# SAD/Grad/Conn are divided by 1000 per image and then averaged, matching PaddleSeg.


def _gauss_filter(sigma: float, epsilon: float = 1e-2) -> Tuple[np.ndarray, np.ndarray]:
    half = np.ceil(sigma * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * sigma * epsilon)))
    size = int(2 * half + 1)
    idx = np.arange(size) - half
    gauss = np.exp(-(idx**2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
    dgauss = -idx * gauss / sigma**2
    filter_x = gauss[:, None] * dgauss[None, :]
    filter_x /= np.sqrt((filter_x**2).sum())
    return filter_x, filter_x.T


def _gauss_gradient(img: np.ndarray, sigma: float) -> np.ndarray:
    fx, fy = _gauss_filter(sigma)
    gx = cv2.filter2D(img, -1, fx, borderType=cv2.BORDER_REPLICATE)
    gy = cv2.filter2D(img, -1, fy, borderType=cv2.BORDER_REPLICATE)
    return np.sqrt(gx**2 + gy**2)


def _minmax(img: np.ndarray) -> np.ndarray:
    lo, hi = float(img.min()), float(img.max())
    if hi - lo < 1e-12:
        return np.zeros_like(img)
    return (img - lo) / (hi - lo)


def sad(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.abs(pred - gt).sum()) / 1000.0


def mse(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(((pred - gt) ** 2).sum()) / float(gt.size)


def grad(pred: np.ndarray, gt: np.ndarray, sigma: float = 1.4) -> float:
    gt_grad = _gauss_gradient(_minmax(gt.astype(np.float64)), sigma).astype(np.float32)
    pred_grad = _gauss_gradient(_minmax(pred.astype(np.float64)), sigma).astype(np.float32)
    return float(((gt_grad - pred_grad) ** 2).sum()) / 1000.0


def conn(pred: np.ndarray, gt: np.ndarray, step: float = 0.1) -> float:
    thresholds = np.arange(0, 1 + step, step)
    round_down = -np.ones_like(gt)
    for i in range(1, len(thresholds)):
        intersection = ((gt >= thresholds[i]) & (pred >= thresholds[i])).astype(np.uint8)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(intersection, connectivity=4)
        omega = np.zeros_like(gt)
        sizes = stats[1:, -1]
        if sizes.size:
            omega[labels == int(np.argmax(sizes)) + 1] = 1
        round_down[(round_down == -1) & (omega == 0)] = thresholds[i - 1]
    round_down[round_down == -1] = 1

    gt_diff, pred_diff = gt - round_down, pred - round_down
    gt_phi = 1 - gt_diff * (gt_diff >= 0.15)
    pred_phi = 1 - pred_diff * (pred_diff >= 0.15)
    return float(np.abs(gt_phi - pred_phi).sum()) / 1000.0


@EVALUATOR_REGISTRY.register("image_matting")
class ImageMattingEvaluator(EvaluatorBase):
    """Image matting evaluator reporting SAD / MSE / Grad / Conn.

    All four metrics are error measures: lower is better.
    """

    def __init__(self, session: SessionBase, dataset: DatasetBase, **kwargs) -> None:
        super().__init__(session, dataset, workers=12, **kwargs)

    def init_metrics(self) -> dict:
        return {"sad": 0.0, "mse": 0.0, "grad": 0.0, "conn": 0.0, "count": 0}

    def extract_inputs(self, batch_data: Tuple) -> np.ndarray:
        image, _ = batch_data
        return image

    def process_batch_result(self, batch_data: Tuple, output: np.ndarray, metrics_state: dict) -> dict:
        _, alpha = batch_data
        gt = np.asarray(alpha, dtype=np.float32).squeeze()
        pred = np.asarray(output, dtype=np.float32).squeeze()
        if pred.ndim != 2:
            raise ValueError(f"Matting prediction must be a single alpha map, got shape {pred.shape}")
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
        pred = np.clip(pred, 0.0, 1.0)

        metrics_state["sad"] += sad(pred, gt)
        metrics_state["mse"] += mse(pred, gt)
        metrics_state["grad"] += grad(pred, gt)
        metrics_state["conn"] += conn(pred, gt)
        metrics_state["count"] += 1
        return metrics_state

    def compute_final_metrics(self, metrics_state: dict) -> dict:
        count = max(metrics_state["count"], 1)
        avg_fps = len(self.dataset) / self.total_inference_time if self.total_inference_time > 0 else 0.0
        return self._finalize(
            metric_names=["SAD", "MSE", "Grad", "Conn"],
            metric_values=[metrics_state[k] / count for k in ("sad", "mse", "grad", "conn")],
            fps=avg_fps,
        )

    def format_progress_desc(self, metrics_state: dict, current_fps: float) -> str:
        count = max(metrics_state["count"], 1)
        return (
            f"{self.dataset.__class__.__name__} | SAD:{metrics_state['sad'] / count:.2f} |"
            f" Current_FPS:{current_fps:.1f}"
        )
