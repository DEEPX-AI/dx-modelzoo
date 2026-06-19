from __future__ import annotations

from typing import Any, Tuple

import cv2
import numpy as np

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.evaluator import EVALUATOR_REGISTRY, EvaluatorBase
from dx_modelzoo.session import SessionBase


def calculate_psnr(img1: np.ndarray, img2: np.ndarray, max_value: float = 255.0) -> float:
    img1_255 = img1 * max_value
    img2_255 = img2 * max_value
    img1_255 = img1_255.astype(np.float64)
    img2_255 = img2_255.astype(np.float64)
    mse = np.mean((img1_255 - img2_255) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10((max_value**2) / (mse + 1e-8))


def calculate_ssim(img1: np.ndarray, img2: np.ndarray, max_value: float = 255.0) -> float:
    img1 = (img1 * max_value).astype(np.float64)
    img2 = (img2 * max_value).astype(np.float64)
    c1 = (0.01 * max_value) ** 2
    c2 = (0.03 * max_value) ** 2
    img_size = min(img1.shape[0], img1.shape[1])
    if img_size < 50:
        kernel_size = 5
        border = 2
    else:
        kernel_size = 11
        border = 5
    kernel = cv2.getGaussianKernel(kernel_size, 1.5)
    kernel_window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, kernel_window)[border:-border, border:-border]
    mu2 = cv2.filter2D(img2, -1, kernel_window)[border:-border, border:-border]
    if mu1.size == 0 or mu2.size == 0:
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return 1.0
        mu = max_value * max_value
        return float((2 * mu + c1) / (mu + mu + c1 + mse))
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, kernel_window)[border:-border, border:-border] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, kernel_window)[border:-border, border:-border] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, kernel_window)[border:-border, border:-border] - mu1_mu2
    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    denominator = np.where(denominator == 0, 1e-8, denominator)
    ssim_map = numerator / denominator
    return float(np.mean(ssim_map))


@EVALUATOR_REGISTRY.register("super_resolution")
class SuperResolutionEvaluator(EvaluatorBase):
    """Super Resolution Evaluator (PSNR/SSIM)."""

    def __init__(self, session: SessionBase, dataset: DatasetBase, **kwargs) -> None:
        super().__init__(session, dataset, **kwargs)
        self.psnr_values = []
        self.ssim_values = []
        self._upscale_factor = None

    @property
    def upscale_factor(self) -> int:
        if self._upscale_factor is None:
            raise ValueError("upscale_factor property is not set.")
        return self._upscale_factor

    @upscale_factor.setter
    def upscale_factor(self, upscale_factor: int) -> None:
        self._upscale_factor = upscale_factor

    def init_metrics(self) -> dict:
        return {"total_psnr": 0.0, "total_ssim": 0.0, "total_samples": 0}

    def extract_inputs(self, batch_data: Tuple) -> np.ndarray:
        lr_images, hr_images = batch_data
        return lr_images

    def process_batch_result(self, batch_data: Tuple, output: Any, metrics_state: dict) -> dict:
        lr_images, hr_images = batch_data
        sr_img = np.asarray(output)
        # squeeze batch dim: (1,C,H,W) → (C,H,W), then CHW → HWC
        if sr_img.ndim == 4:
            sr_img = sr_img[0]
        if sr_img.ndim == 3 and sr_img.shape[0] in (1, 3):
            sr_img = sr_img.transpose(1, 2, 0)
        if sr_img.ndim == 3 and sr_img.shape[2] == 1:
            sr_img = sr_img[:, :, 0]
        hr_img = np.asarray(hr_images)

        # Ensure HWC format for both
        if hr_img.ndim == 4:
            hr_img = hr_img[0]
        if hr_img.ndim == 3 and hr_img.shape[0] in (1, 3):
            hr_img = hr_img.transpose(1, 2, 0)
        if hr_img.ndim == 3 and hr_img.shape[2] == 1:
            hr_img = hr_img[:, :, 0]
        if hr_img.ndim == 2 and sr_img.ndim == 2:
            pass  # both grayscale 2D, OK
        elif hr_img.ndim == 2:
            hr_img = hr_img[..., np.newaxis]

        # Resize HR to match SR if needed
        sr_h, sr_w = sr_img.shape[:2]
        hr_h, hr_w = hr_img.shape[:2]
        if sr_h != hr_h or sr_w != hr_w:
            hr_img = cv2.resize(hr_img, (sr_w, sr_h))

        # Ensure matching dims for metric calculation
        if sr_img.ndim == 2 and hr_img.ndim == 3:
            hr_img = hr_img[:, :, 0] if hr_img.shape[2] == 1 else hr_img
        elif sr_img.ndim == 3 and hr_img.ndim == 2:
            hr_img = hr_img[..., np.newaxis]

        psnr = calculate_psnr(sr_img, hr_img)
        ssim = calculate_ssim(sr_img, hr_img)
        metrics_state["total_psnr"] += psnr
        metrics_state["total_ssim"] += ssim
        metrics_state["total_samples"] += 1
        return metrics_state

    def compute_final_metrics(self, metrics_state: dict) -> dict:
        total = metrics_state["total_samples"]
        final_psnr = metrics_state["total_psnr"] / total if total > 0 else 0.0
        final_ssim = metrics_state["total_ssim"] / total if total > 0 else 0.0
        avg_fps = total / self.total_inference_time if self.total_inference_time > 0 else 0.0
        return self._finalize(
            metric_names=["PSNR", "SSIM"],
            metric_values=[final_psnr, final_ssim],
            fps=avg_fps,
        )

    def format_progress_desc(self, metrics_state: dict, current_fps: float) -> str:
        total = metrics_state["total_samples"]
        if total == 0:
            return "SR Eval | Initializing..."
        avg_psnr = metrics_state["total_psnr"] / total
        avg_ssim = metrics_state["total_ssim"] / total
        return f"SR Eval | PSNR: {avg_psnr:.2f}dB | SSIM: {avg_ssim:.4f} | Current_FPS: {current_fps:.1f}"
