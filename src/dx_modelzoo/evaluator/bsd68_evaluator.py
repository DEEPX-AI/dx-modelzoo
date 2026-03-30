import math
from typing import List, Tuple

import cv2
import numpy as np
import torch
from loguru import logger

from dx_modelzoo.enums import SessionType
from dx_modelzoo.evaluator import EvaluatorBase


class BSD68Evaluator(EvaluatorBase):
    """BSD68 Evaluator for Image Denosing.

    Args:
        session: runtime session.
        dataset: COCO dataset.
    """

    def __init__(self, session, dataset):
        super().__init__(session, dataset, workers=12)

        self._noise_level = None
        self._input_size = None
        self._use_color = False

    @property
    def noise_level(self) -> int:
        if self._noise_level is None:
            raise ValueError("noise_level property is not set.")
        return self._noise_level

    @property
    def input_size(self) -> Tuple[int, int]:
        if self._input_size is None:
            raise ValueError("input_size property is not set.")
        return self._input_size

    @property
    def use_color(self) -> bool:
        return self._use_color

    @noise_level.setter
    def noise_level(self, noise_level):
        self._noise_level = noise_level

    @input_size.setter
    def input_size(self, input_size):
        self._input_size = input_size

    @use_color.setter
    def use_color(self, use_color: bool):
        self._use_color = use_color

    def init_metrics(self) -> dict:
        """Initialize metrics state."""
        np.random.seed(seed=0)  # Set seed for reproducibility
        return {
            "psnr_list": [],
            "ssim_list": [],
        }

    def extract_inputs(self, batch_data: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Extract and preprocess image from batch data."""
        inp_image, origin_image = batch_data
        img_shape = inp_image.shape
        inp_image = self.perturb_image(inp_image)
        inp_image = self.padding(inp_image, img_shape)
        return inp_image

    def process_batch_result(
        self,
        batch_data: Tuple[torch.Tensor, torch.Tensor],
        output: np.ndarray,
        metrics_state: dict,
    ) -> dict:
        """Process batch result and update PSNR/SSIM metrics."""
        inp_image, origin_image = batch_data
        img_shape = inp_image.shape

        if self.session.type == SessionType.dxruntime and self.use_color:
            _, h, w, _ = img_shape
        else:
            _, _, h, w = img_shape

        output = np.squeeze(output)

        if self.use_color:
            output = np.transpose(output, (1, 2, 0))

        output = np.uint8((output * 255.0).clip(0, 255).round())
        output = output[:h, :w]

        origin_image = np.squeeze(origin_image.cpu().detach().numpy())

        if self.use_color:
            origin_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)
        else:
            origin_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2GRAY)

        psnr = self.calculate_psnr(origin_image, output)
        ssim = self.calculate_ssim(origin_image, output)

        metrics_state["psnr_list"].append(psnr)
        metrics_state["ssim_list"].append(ssim)

        return metrics_state

    def compute_final_metrics(self, metrics_state: dict) -> dict:
        """Compute final PSNR and SSIM metrics."""
        psnr_list = metrics_state["psnr_list"]
        ssim_list = metrics_state["ssim_list"]

        avg_psnr = sum(psnr_list) / len(psnr_list) if psnr_list else 0.0
        avg_ssim = sum(ssim_list) / len(ssim_list) if ssim_list else 0.0
        avg_fps = len(psnr_list) / self.total_inference_time if self.total_inference_time > 0 else 0.0

        print(f"Noise Level: {self._noise_level}")
        print(f"Average PSNR, Average SSIM: {avg_psnr:.4f}, {avg_ssim:.4f}")
        print(f"Average FPS: {avg_fps:.2f}")
        logger.success(f"@JSON <Average PSNR:{avg_psnr:.4f}; Average SSIM:{avg_ssim:.4f}; Average FPS:{avg_fps:.2f}>")

        return {
            "performance": [avg_psnr, avg_ssim],
            "fps": avg_fps,
        }

    def format_progress_desc(self, metrics_state: dict, current_fps: float) -> str:
        """Format progress bar description."""
        return f"BSD68 | Current_FPS:{current_fps:.1f}"

    def perturb_image(self, img: torch.Tensor) -> torch.Tensor:
        """make input noisy image."""
        if self.session.type == SessionType.dxruntime:
            img = img.type(torch.float32) / 255.0
        img += np.random.normal(0, self._noise_level / 255.0, img.shape)
        if self.session.type == SessionType.dxruntime:
            img *= 255
            img = img.clamp(0, 255).round().type(torch.uint8)
        return img

    def padding(self, img: torch.Tensor, img_shape: List[int]) -> torch.Tensor:
        if self.session.type == SessionType.dxruntime and self.use_color:
            _, h, w, _ = img_shape
            pad_h, pad_w = self.input_size[0] - h, self.input_size[1] - w
            return torch.nn.functional.pad(img, (0, 0, 0, pad_w, 0, pad_h))
        else:
            _, _, h, w = img_shape
            pad_h, pad_w = self.input_size[0] - h, self.input_size[1] - w
            return torch.nn.functional.pad(img, (0, pad_w, 0, pad_h))

    def calculate_psnr(self, img1: np.ndarray, img2: np.ndarray, boarder: int = 0) -> float:
        if img1.shape != img2.shape:
            raise ValueError("Input images must have the same dimensions.")
        h, w = img1.shape[:2]
        img1 = img1[boarder : h - boarder, boarder : w - boarder].astype(np.float32)
        img2 = img2[boarder : h - boarder, boarder : w - boarder].astype(np.float32)
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float("inf")
        return 20 * math.log10(255.0 / math.sqrt(mse))

    def calculate_ssim(self, img1: np.ndarray, img2: np.ndarray, border=0) -> np.ndarray:
        """calculate SSIM
        the same outputs as MATLAB's
        img1, img2: [0, 255]
        """
        if img1.shape != img2.shape:
            raise ValueError("Input images must have the same dimensions.")
        h, w = img1.shape[:2]
        img1 = img1[border : h - border, border : w - border]
        img2 = img2[border : h - border, border : w - border]

        if img1.ndim == 2:
            return self.ssim(img1, img2)
        elif img1.ndim == 3:
            if img1.shape[2] == 3:
                ssims = []
                for i in range(3):
                    ssims.append(self.ssim(img1[:, :, i], img2[:, :, i]))
                return np.array(ssims).mean()
            elif img1.shape[2] == 1:
                return self.ssim(np.squeeze(img1), np.squeeze(img2))
        else:
            raise ValueError("Wrong input image dimensions.")

    def ssim(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()
