from typing import Tuple

import cv2
import numpy as np
import torch
from loguru import logger

from dx_modelzoo.dataset import DatasetBase
from dx_modelzoo.evaluator import EvaluatorBase
from dx_modelzoo.session import SessionBase
from dx_modelzoo.utils import torch_to_numpy


def calculate_psnr(img1: np.ndarray, img2: np.ndarray, max_value: float = 255.0) -> float:
    """Calculate PSNR between two images following original ESPCN implementation.

    Args:
        img1 (np.ndarray): First image in [0, 1] range
        img2 (np.ndarray): Second image in [0, 1] range
        max_value (float): Maximum possible pixel value (default: 255.0)

    Returns:
        float: PSNR value in dB
    """
    img1_255 = img1 * 255.0
    img2_255 = img2 * 255.0

    img1_255 = img1_255.astype(np.float64)
    img2_255 = img2_255.astype(np.float64)

    mse = np.mean((img1_255 - img2_255) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10((255.0**2) / (mse + 1e-8))


def calculate_ssim(img1: np.ndarray, img2: np.ndarray, max_value: float = 255.0) -> float:
    """Calculate SSIM between two images.

    Follows the original ESPCN implementation from image_quality_assessment.py

    Args:
        img1 (np.ndarray): First image in [0, 1] range
        img2 (np.ndarray): Second image in [0, 1] range
        max_value (float): Maximum possible pixel value (default: 255.0 for calculation)

    Returns:
        float: SSIM value (0-1)
    """
    img1 = (img1 * 255.0).astype(np.float64)
    img2 = (img2 * 255.0).astype(np.float64)

    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2

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
        max_val = 255.0
        return (2 * max_val * max_val + c1) / (max_val * max_val + max_val * max_val + c1 + mse)

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


class BSD100Evaluator(EvaluatorBase):
    """Super Resolution Evaluator for ESPCN and other SR models.

    Evaluates super resolution models using PSNR and SSIM metrics.

    Args:
        session: Runtime session (ONNX or DX runtime)
        dataset: Super resolution dataset (e.g., BSD100Dataset)

    Example:
        evaluator = SuperResolutionEvaluator(session, dataset)
        evaluator.eval()

        evaluator = SuperResolutionEvaluator(session, dataset)
        evaluator.eval()
    """

    def __init__(self, session: SessionBase, dataset: DatasetBase) -> None:
        super().__init__(session, dataset)

        self.psnr_values = []
        self.ssim_values = []

        self.image_counter = 0

        self._upscale_factor = None

    @property
    def upscale_factor(self) -> int:
        if self._upscale_factor is None:
            raise ValueError("upscale_factor property is not set.")
        return self._upscale_factor

    @upscale_factor.setter
    def upscale_factor(self, upscale_factor):
        self._upscale_factor = upscale_factor
        self.dataset.data_dir = self.dataset.data_dir + f"/bicubic_{str(upscale_factor)}x"

    def init_metrics(self) -> dict:
        """Initialize metrics state."""
        return {
            "total_psnr": 0.0,
            "total_ssim": 0.0,
            "total_samples": 0,
        }

    def extract_inputs(self, batch_data: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Extract LR image from batch data."""
        lr_images, hr_images = batch_data
        return lr_images

    def process_batch_result(
        self,
        batch_data: Tuple[torch.Tensor, torch.Tensor],
        output: torch.Tensor,
        metrics_state: dict,
    ) -> dict:
        """Process batch result and update PSNR/SSIM metrics."""
        lr_images, hr_images = batch_data
        batch_size = lr_images.size(0)

        sr_outputs = output[0]

        if isinstance(sr_outputs, torch.Tensor):
            sr_outputs = torch_to_numpy(sr_outputs)

        hr_numpy = torch_to_numpy(hr_images)

        batch_psnr_sum = 0.0
        batch_ssim_sum = 0.0

        for i in range(batch_size):
            sr_img = sr_outputs[i]
            hr_img = hr_numpy[i]

            sr_shape = sr_img.shape
            hr_shape = hr_img.shape

            if len(sr_shape) >= 2 and len(hr_shape) >= 2:
                sr_h, sr_w = sr_shape[-2], sr_shape[-1]
                hr_h, hr_w = hr_shape[-2], hr_shape[-1]

                if sr_h != hr_h or sr_w != hr_w:
                    if len(hr_shape) == 3:
                        hr_resized = np.zeros((hr_shape[0], sr_h, sr_w), dtype=hr_img.dtype)
                        for c in range(hr_shape[0]):
                            hr_resized[c] = cv2.resize(hr_img[c], (sr_w, sr_h), interpolation=cv2.INTER_LINEAR)
                        hr_img = hr_resized
                    else:
                        hr_img = cv2.resize(hr_img, (sr_w, sr_h), interpolation=cv2.INTER_LINEAR)

            psnr = calculate_psnr(sr_img, hr_img)
            ssim = calculate_ssim(sr_img, hr_img)

            batch_psnr_sum += psnr
            batch_ssim_sum += ssim

            self.psnr_values.append(psnr)
            self.ssim_values.append(ssim)

        metrics_state["total_psnr"] += batch_psnr_sum
        metrics_state["total_ssim"] += batch_ssim_sum
        metrics_state["total_samples"] += batch_size

        return metrics_state

    def compute_final_metrics(self, metrics_state: dict) -> dict:
        """Compute final PSNR and SSIM metrics."""
        total_psnr = metrics_state["total_psnr"]
        total_ssim = metrics_state["total_ssim"]
        total_samples = metrics_state["total_samples"]

        final_psnr = total_psnr / total_samples if total_samples > 0 else 0.0
        final_ssim = total_ssim / total_samples if total_samples > 0 else 0.0
        avg_fps = total_samples / self.total_inference_time if self.total_inference_time > 0 else 0.0

        print(f"Upscale Factor: {self._upscale_factor}")
        print(f"Average PSNR, Average SSIM: {final_psnr:.4f}, {final_ssim:.6f}")
        print(f"Average FPS: {avg_fps:.2f}")
        logger.success(f"@JSON <PSNR:{final_psnr:.4f}; SSIM:{final_ssim:.6f}; Average FPS:{avg_fps:.2f}>")

        return {
            "performance": [final_psnr, final_ssim],
            "fps": avg_fps,
        }

    def format_progress_desc(self, metrics_state: dict, current_fps: float) -> str:
        """Format progress bar description."""
        total_psnr = metrics_state["total_psnr"]
        total_ssim = metrics_state["total_ssim"]
        total_samples = metrics_state["total_samples"]

        if total_samples == 0:
            return "SR Eval | Initializing..."

        avg_psnr = total_psnr / total_samples
        avg_ssim = total_ssim / total_samples

        return f"SR Eval | PSNR: {avg_psnr:.2f}dB | SSIM: {avg_ssim:.4f} | Current_FPS: {current_fps:.1f}"

    def get_detailed_results(self) -> dict:
        """Get detailed evaluation results.

        Returns:
            dict: Dictionary containing detailed metrics
        """
        if not self.psnr_values or not self.ssim_values:
            return {"error": "No evaluation results available"}

        return {
            "psnr": {
                "mean": np.mean(self.psnr_values),
                "std": np.std(self.psnr_values),
                "min": np.min(self.psnr_values),
                "max": np.max(self.psnr_values),
            },
            "ssim": {
                "mean": np.mean(self.ssim_values),
                "std": np.std(self.ssim_values),
                "min": np.min(self.ssim_values),
                "max": np.max(self.ssim_values),
            },
            "total_samples": len(self.psnr_values),
            "avg_inference_time_ms": (self.total_inference_time / len(self.psnr_values) * 1000)
            if self.psnr_values
            else 0,
        }
