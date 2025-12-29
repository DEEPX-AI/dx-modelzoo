import math
import os
import time
from collections import deque
from typing import List, Tuple

import cv2
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

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
        return float('inf')
    return 10 * np.log10((255.0 ** 2) / (mse + 1e-8))


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
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(img1 ** 2, -1, kernel_window)[border:-border, border:-border] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, kernel_window)[border:-border, border:-border] - mu2_sq
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
        
        self.total_inference_time = 0.0
        self.recent_inference_times = deque(maxlen=100)
        
        self.psnr_values = []
        self.ssim_values = []
        
        self.image_counter = 0

    def eval(self) -> None:
        """Evaluate super resolution model using PSNR and SSIM metrics."""
        loader = self.make_loader()
        total_len = len(loader)
        
        total_psnr = 0.0
        total_ssim = 0.0
        total_samples = 0

        pbar = tqdm(enumerate(loader), total=total_len, desc="Super Resolution Evaluation")
        
        for batch_idx, (lr_images, hr_images) in pbar:
            batch_size = lr_images.size(0)
            
            batch_psnr, batch_ssim = self._run_one_batch(lr_images, hr_images)
            
            total_psnr += batch_psnr
            total_ssim += batch_ssim
            total_samples += batch_size
            
            avg_psnr = total_psnr / total_samples
            avg_ssim = total_ssim / total_samples
            
            if len(self.recent_inference_times) > 0:
                current_fps = len(self.recent_inference_times) / sum(self.recent_inference_times)
            else:
                current_fps = 0.0
            
            pbar.set_description(
                f"SR Eval | PSNR: {avg_psnr:.2f}dB | SSIM: {avg_ssim:.4f} | FPS: {current_fps:.1f}"
            )

        final_psnr = total_psnr / total_samples
        final_ssim = total_ssim / total_samples
        avg_fps = total_samples / self.total_inference_time if self.total_inference_time > 0 else 0.0
        
        logger.info(f"Super Resolution Evaluation Results:")
        logger.info(f"  Average PSNR: {final_psnr:.4f} dB")
        logger.info(f"  Average SSIM: {final_ssim:.6f}")
        logger.info(f"  Total samples: {total_samples}")
        
        if self.total_inference_time > 0:
            avg_inference_time = self.total_inference_time / total_samples
            logger.info(f"  Average inference time: {avg_inference_time*1000:.2f} ms")

        return {
            "performance": [final_psnr, final_ssim], 
            "fps": avg_fps,
        }

    def _run_one_batch(self, lr_images, hr_images) -> Tuple[float, float]:
        """Run inference on one batch and calculate metrics.
        
        Args:
            lr_images: Preprocessed LR input images (tensor)
            hr_images: Preprocessed HR ground truth images (tensor)
            
        Returns:
            Tuple[float, float]: (batch_psnr_sum, batch_ssim_sum)
        """
        batch_size = lr_images.size(0)
        start_time = time.time()
        sr_outputs = self.session.run(lr_images)[0]
        inference_time = time.time() - start_time
        
        self.total_inference_time += inference_time
        self.recent_inference_times.append(inference_time)

        if isinstance(sr_outputs, torch.Tensor):
            sr_outputs = torch_to_numpy(sr_outputs)
        
        hr_numpy = torch_to_numpy(hr_images)
        
        batch_psnr_sum = 0.0
        batch_ssim_sum = 0.0
        
        lr_numpy = torch_to_numpy(lr_images)
        
        for i in range(batch_size):
            lr_img = lr_numpy[i]
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
            
        
        return batch_psnr_sum, batch_ssim_sum

        
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
                "max": np.max(self.psnr_values)
            },
            "ssim": {
                "mean": np.mean(self.ssim_values),
                "std": np.std(self.ssim_values),
                "min": np.min(self.ssim_values),
                "max": np.max(self.ssim_values)
            },
            "total_samples": len(self.psnr_values),
            "avg_inference_time_ms": (self.total_inference_time / len(self.psnr_values) * 1000) if self.psnr_values else 0
        }