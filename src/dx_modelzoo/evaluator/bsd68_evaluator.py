import math
import queue
import threading
import time
from collections import deque
from typing import List, Tuple

import cv2
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from dx_modelzoo.enums import SessionType
from dx_modelzoo.evaluator import EvaluatorBase


class BSD68Evaluator(EvaluatorBase):
    """BSD68 Evaluator for Image Denosing.

    Args:
        session: runtime session.
        dataset: COCO dataset.
    """

    def __init__(self, session, dataset, async_queue_size: int = 32):
        super().__init__(session, dataset)

        self._noise_level = None
        self._input_size = None
        self.async_queue_size = async_queue_size
        self._use_async = session.type == SessionType.dxruntime

    @property
    def nosie_level(self) -> int:
        if self._noise_level is None:
            raise ValueError("noise_level property is not set.")
        return self._noise_level

    @property
    def input_size(self) -> Tuple[int, int]:
        if self._input_size is None:
            raise ValueError("input_size property is not set.")
        return self._input_size

    @nosie_level.setter
    def noise_level(self, noise_level):
        self._noise_level = noise_level

    @input_size.setter
    def input_size(self, input_size):
        self._input_size = input_size

    def eval(self):
        if self._use_async:
            logger.info(f"Using async evaluation with queue_size={self.async_queue_size}")
            return self._eval_async()
        return self._eval_sync()

    def _eval_sync(self):
        """Synchronous evaluation (original behavior)."""
        loader = self.make_loader()
        total_len = len(loader)
        np.random.seed(seed=0)

        psnr_list = []
        ssim_list = []

        recent_inference_times = deque(maxlen=5)  # total : 68
        total_inference_time = 0.0

        pbar = tqdm(loader, total=total_len)
        for inp_image, origin_image in pbar:
            img_shape = inp_image.shape
            inp_image = self.perturb_image(inp_image)
            inp_image = self.padding(inp_image, img_shape)

            start_time = time.time()
            outputs = self.session.run(inp_image)
            inference_time = time.time() - start_time

            recent_inference_times.append(inference_time)
            total_inference_time += inference_time

            outputs = self.postprocessing(outputs)

            _, _, h, w = img_shape
            outputs = np.squeeze(outputs)
            outputs = np.uint8((outputs * 255.0).clip(0, 255).round())
            outputs = outputs[:h, :w]

            origin_image = np.squeeze(origin_image.cpu().detach().numpy())
            origin_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2GRAY)

            psnr_list.append(self.calculate_psnr(origin_image, outputs))
            ssim_list.append(self.calculate_ssim(origin_image, outputs))

            if len(recent_inference_times) > 0:
                current_fps = len(recent_inference_times) / sum(recent_inference_times)
            else:
                current_fps = 0.0

            pbar.desc = f"BSD68 | Current_FPS:{current_fps:.1f}"

        return self._finalize_results(psnr_list, ssim_list, total_len, total_inference_time)

    def _eval_async(self):
        """Asynchronous evaluation for NPU session using native async API."""
        loader = self.make_loader()
        total_len = len(loader)
        np.random.seed(seed=0)  # Must be set before perturb_image calls

        # Thread-safe state
        job_queue: queue.Queue = queue.Queue(maxsize=self.async_queue_size)
        result_lock = threading.Lock()

        psnr_list = []
        ssim_list = []
        processed_count = 0
        worker_done = threading.Event()

        wall_start_time = time.time()
        pbar = tqdm(total=total_len)

        def worker_thread():
            """Worker thread: wait for jobs and process results."""
            nonlocal processed_count

            while True:
                try:
                    item = job_queue.get(timeout=1.0)
                except queue.Empty:
                    if worker_done.is_set() and job_queue.empty():
                        break
                    continue

                if item is None:  # Poison pill
                    break

                job_id, origin_image, img_shape = item

                # Wait for NPU to complete
                outputs = self.session.wait(job_id)

                # Postprocessing
                outputs = self.postprocessing(outputs)

                _, _, h, w = img_shape
                outputs = np.squeeze(outputs)
                outputs = np.uint8((outputs * 255.0).clip(0, 255).round())
                outputs = outputs[:h, :w]

                origin_np = np.squeeze(origin_image.cpu().detach().numpy())
                origin_np = cv2.cvtColor(origin_np, cv2.COLOR_BGR2GRAY)

                psnr = self.calculate_psnr(origin_np, outputs)
                ssim = self.calculate_ssim(origin_np, outputs)

                with result_lock:
                    psnr_list.append(psnr)
                    ssim_list.append(ssim)
                    processed_count += 1

                    elapsed = time.time() - wall_start_time
                    current_fps = processed_count / elapsed if elapsed > 0 else 0.0

                    pbar.desc = f"BSD68 | Current_FPS:{current_fps:.1f}"
                    pbar.update(1)

                job_queue.task_done()

        # Start worker thread
        worker = threading.Thread(target=worker_thread, daemon=True)
        worker.start()

        # Main thread: perturb images in order and submit async jobs
        for inp_image, origin_image in loader:
            img_shape = inp_image.shape
            inp_image = self.perturb_image(inp_image)  # Must be in order for reproducibility
            inp_image = self.padding(inp_image, img_shape)

            job_id = self.session.run_async(inp_image)
            job_queue.put((job_id, origin_image, img_shape))

        # Signal completion and wait for worker
        worker_done.set()
        job_queue.put(None)
        worker.join()

        pbar.close()

        total_inference_time = time.time() - wall_start_time
        return self._finalize_results(psnr_list, ssim_list, total_len, total_inference_time)

    def _finalize_results(
        self, psnr_list: List[float], ssim_list: List[float], total_len: int, total_time: float
    ) -> dict:
        """Calculate and log final metrics."""
        avg_psnr = sum(psnr_list) / len(psnr_list)
        avg_ssim = sum(ssim_list) / len(ssim_list)
        avg_fps = total_len / total_time if total_time > 0 else 0

        print(f"Noise Level: {self.noise_level}")
        print(f"Average PSNR, Average SSIM: {avg_psnr:.4f}, {avg_ssim:.4f}")
        print(f"Average FPS: {avg_fps:.2f}")
        logger.success(f"@JSON <Average PSNR:{avg_psnr:.4f}; Average SSIM:{avg_ssim:.4f}; Average FPS:{avg_fps:.2f}>")

        return {
            "performance": [avg_psnr, avg_ssim],
            "fps": avg_fps,
        }

    def perturb_image(self, img: torch.Tensor) -> torch.Tensor:
        """make input noisy image."""
        if self.session.type == SessionType.dxruntime:
            img = img.type(torch.float32) / 255.0
        img += np.random.normal(0, self.noise_level / 255.0, img.shape)
        if self.session.type == SessionType.dxruntime:
            img *= 255
            img = img.clamp(0, 255).round().type(torch.uint8)
        return img

    def padding(self, img: torch.Tensor, img_shape: List[int]) -> torch.Tensor:
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
