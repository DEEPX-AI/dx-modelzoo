from __future__ import annotations

import math
from typing import Tuple

import cv2
import numpy as np

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.evaluator import EVALUATOR_REGISTRY, EvaluatorBase
from dx_modelzoo.session import SessionBase


@EVALUATOR_REGISTRY.register("image_denoising")
class ImageDenoisingEvaluator(EvaluatorBase):
    """BSD68 Evaluator for Image Denoising (PSNR/SSIM)."""

    def __init__(self, session: SessionBase, dataset: DatasetBase, **kwargs) -> None:
        super().__init__(session, dataset, workers=12, **kwargs)
        self._noise_level = None
        self._input_size = (512, 512)
        self._use_color = False
        self.is_npu = False

    @property
    def noise_level(self) -> int:
        if self._noise_level is None:
            raise ValueError("noise_level property is not set.")
        return self._noise_level

    @noise_level.setter
    def noise_level(self, noise_level: int) -> None:
        self._noise_level = noise_level

    @property
    def input_size(self) -> Tuple[int, int]:
        if self._input_size is None:
            raise ValueError("input_size property is not set.")
        return self._input_size

    @input_size.setter
    def input_size(self, input_size: Tuple[int, int]) -> None:
        self._input_size = input_size

    @property
    def use_color(self) -> bool:
        return self._use_color

    @use_color.setter
    def use_color(self, use_color: bool) -> None:
        self._use_color = use_color

    def init_metrics(self) -> dict:
        np.random.seed(seed=0)
        return {"psnr_list": [], "ssim_list": []}

    def _ensure_4d(self, inp_image: np.ndarray) -> np.ndarray:
        """Ensure image is 4D for evaluator shape unpacking.

        Grayscale target: (H,W) or (H,W,1) or (1,H,W) → (1,1,H,W) NCHW
        Color target: (3,H,W) or (H,W,3) → (1,H,W,3) NHWC
        """
        if not self.use_color:
            if inp_image.ndim == 3 and inp_image.shape[-1] == 1:
                inp_image = inp_image[:, :, 0]
            while inp_image.ndim < 4:
                inp_image = np.expand_dims(inp_image, axis=0)
        else:
            if inp_image.ndim == 3 and inp_image.shape[0] in (1, 3):
                inp_image = np.transpose(inp_image, (1, 2, 0))  # CHW → HWC
            while inp_image.ndim < 4:
                inp_image = np.expand_dims(inp_image, axis=0)
        return inp_image

    def extract_inputs(self, batch_data: Tuple) -> np.ndarray:
        inp_image, origin_image = batch_data
        inp_image = self._ensure_4d(inp_image)
        img_shape = inp_image.shape
        inp_image = self.perturb_image(inp_image)
        inp_image = self.padding(inp_image, img_shape)
        return inp_image

    def _build_postprocessing_context(self, batch_data) -> dict:
        inp_image, _ = batch_data
        inp_image = self._ensure_4d(inp_image)
        img_shape = inp_image.shape
        if self.use_color and self.is_npu:
            _, h, w, _ = img_shape
        else:
            _, _, h, w = img_shape
        return {"content_h": h, "content_w": w, "use_color": self.use_color}

    def process_batch_result(self, batch_data: Tuple, output: np.ndarray, metrics_state: dict) -> dict:
        """Accumulate PSNR/SSIM metrics.

        ``output`` is a uint8 HW or HWC image from postprocessing
        (squeeze, transpose, scale, crop already applied by pp).
        """
        inp_image, origin_image = batch_data
        inp_image = self._ensure_4d(inp_image)
        img_shape = inp_image.shape

        if self.use_color and self.is_npu:
            _, h, w, _ = img_shape
        else:
            _, _, h, w = img_shape

        pred = np.asarray(output).squeeze()
        if self.use_color and pred.ndim == 3 and pred.shape[0] in (1, 3):
            pred = pred.transpose(1, 2, 0)
        pred = np.uint8((pred * 255.0).clip(0, 255).round())
        pred = pred[:h, :w]

        origin_image = np.squeeze(np.asarray(origin_image))
        origin_image = origin_image[:h, :w]
        if pred.ndim == 2 and origin_image.ndim == 3:
            origin_image = np.dot(origin_image[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

        psnr = self.calculate_psnr(origin_image, pred)
        ssim = self.calculate_ssim(origin_image, pred)
        metrics_state["psnr_list"].append(psnr)
        metrics_state["ssim_list"].append(ssim)
        return metrics_state

    def compute_final_metrics(self, metrics_state: dict) -> dict:
        psnr_list = metrics_state["psnr_list"]
        ssim_list = metrics_state["ssim_list"]
        avg_psnr = sum(psnr_list) / len(psnr_list) if psnr_list else 0.0
        avg_ssim = sum(ssim_list) / len(ssim_list) if ssim_list else 0.0
        avg_fps = len(psnr_list) / self.total_inference_time if self.total_inference_time > 0 else 0.0
        return self._finalize(
            metric_names=["PSNR", "SSIM"],
            metric_values=[avg_psnr, avg_ssim],
            fps=avg_fps,
        )

    def format_progress_desc(self, metrics_state: dict, current_fps: float) -> str:
        return f"BSD68 | Current_FPS:{current_fps:.1f}"

    def perturb_image(self, img: np.ndarray) -> np.ndarray:
        if self.is_npu:
            # NPU: input is uint8, normalize → add noise → convert back to uint8
            img = img.astype(np.float32) / 255.0
            img += np.random.normal(0, self._noise_level / 255.0, img.shape).astype(np.float32)
            img = np.uint8((img * 255.0).clip(0, 255).round())
        else:
            # ONNX: input is already float32 0-1 from preprocessing div(255)
            img += np.random.normal(0, self._noise_level / 255.0, img.shape).astype(np.float32)
        return img

    def padding(self, img: np.ndarray, img_shape: Tuple) -> np.ndarray:
        if self.use_color and self.is_npu:
            _, h, w, _ = img_shape
            pad_h, pad_w = self.input_size[0] - h, self.input_size[1] - w
            return np.pad(img, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)))
        else:
            _, _, h, w = img_shape
            pad_h, pad_w = self.input_size[0] - h, self.input_size[1] - w
            return np.pad(img, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)))

    def calculate_psnr(self, img1: np.ndarray, img2: np.ndarray, border: int = 0) -> float:
        if img1.shape != img2.shape:
            raise ValueError("Input images must have the same dimensions.")
        h, w = img1.shape[:2]
        img1 = img1[border : h - border, border : w - border].astype(np.float32) if border else img1.astype(np.float32)
        img2 = img2[border : h - border, border : w - border].astype(np.float32) if border else img2.astype(np.float32)
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float("inf")
        return 20 * math.log10(255.0 / math.sqrt(mse))

    def calculate_ssim(self, img1: np.ndarray, img2: np.ndarray, border: int = 0) -> float:
        if img1.shape != img2.shape:
            raise ValueError("Input images must have the same dimensions.")
        h, w = img1.shape[:2]
        img1 = img1[border : h - border, border : w - border] if border else img1
        img2 = img2[border : h - border, border : w - border] if border else img2
        if img1.ndim == 2:
            return float(self._ssim(img1, img2))
        elif img1.ndim == 3:
            if img1.shape[2] == 3:
                return float(np.mean([self._ssim(img1[:, :, i], img2[:, :, i]) for i in range(3)]))
            elif img1.shape[2] == 1:
                return float(self._ssim(np.squeeze(img1), np.squeeze(img2)))
        raise ValueError("Wrong input image dimensions.")

    def _ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())
        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return float(ssim_map.mean())
