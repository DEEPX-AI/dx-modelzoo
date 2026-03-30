from typing import Tuple

import cv2
import numpy as np
import torch
from loguru import logger

from dx_modelzoo.dataset.lol import LOLDataset
from dx_modelzoo.evaluator import EvaluatorBase
from dx_modelzoo.evaluator.bsd100_evaluator import calculate_psnr, calculate_ssim
from dx_modelzoo.session import SessionBase
from dx_modelzoo.utils import torch_to_numpy


class LOLEvaluator(EvaluatorBase):
    """Low-Light Image Enhancement Evaluator using PSNR and SSIM.

    Compares model-enhanced images against ground truth normal-light images.

    Args:
        session: Runtime session.
        dataset: LOL dataset.
    """

    def __init__(self, session: SessionBase, dataset: LOLDataset) -> None:
        super().__init__(session, dataset)

    def init_metrics(self) -> dict:
        return {
            "total_psnr": 0.0,
            "total_ssim": 0.0,
            "total_samples": 0,
        }

    def extract_inputs(self, batch_data: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        low_images, high_images = batch_data
        return low_images

    def process_batch_result(
        self,
        batch_data: Tuple[torch.Tensor, torch.Tensor],
        output,
        metrics_state: dict,
    ) -> dict:
        low_images, high_images = batch_data

        enhanced = output[0] if isinstance(output, (list, tuple)) else output
        if isinstance(enhanced, torch.Tensor):
            enhanced = torch_to_numpy(enhanced)

        hr_numpy = torch_to_numpy(high_images) if isinstance(high_images, torch.Tensor) else high_images

        batch_size = enhanced.shape[0] if enhanced.ndim == 4 else 1
        if enhanced.ndim == 3:
            enhanced = enhanced[np.newaxis]
        if hr_numpy.ndim == 3:
            hr_numpy = hr_numpy[np.newaxis]

        for i in range(batch_size):
            pred = enhanced[i]  # [C, H, W] float [0,1]
            gt = hr_numpy[i]  # [H, W, C] uint8 from dataset

            # Convert pred from CHW to HWC to match GT
            if pred.ndim == 3 and pred.shape[0] in (1, 3):
                pred = pred.transpose(1, 2, 0)

            # Normalize GT to [0, 1] if needed
            if gt.max() > 1.0:
                gt = gt.astype(np.float32) / 255.0

            pred = np.clip(pred, 0.0, 1.0)

            # Ensure same spatial size (H, W)
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

        print(f"Average PSNR, Average SSIM: {avg_psnr:.4f}, {avg_ssim:.4f}")
        print(f"Average Inference FPS: {avg_fps:.2f}")
        logger.success(f"@JSON <PSNR:{avg_psnr:.4f}; SSIM:{avg_ssim:.4f}; Average FPS:{avg_fps:.2f}>")

        return {
            "performance": [avg_psnr, avg_ssim],
            "fps": avg_fps,
        }

    def format_progress_desc(self, metrics_state: dict, current_fps: float) -> str:
        total = metrics_state["total_samples"]
        if total == 0:
            return "LLIE | Initializing..."
        avg_psnr = metrics_state["total_psnr"] / total
        avg_ssim = metrics_state["total_ssim"] / total
        return f"LLIE | PSNR:{avg_psnr:.2f}dB SSIM:{avg_ssim:.4f} FPS:{current_fps:.1f}"
