import queue
import threading
import time
from collections import deque
from typing import List

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from dx_modelzoo.dataset import DatasetBase
from dx_modelzoo.enums import SessionType
from dx_modelzoo.evaluator import EvaluatorBase
from dx_modelzoo.session import SessionBase
from dx_modelzoo.utils import torch_to_numpy


class ICEvaluator(EvaluatorBase):
    """Image Classification Evaluator."""

    def __init__(
        self,
        session: SessionBase,
        dataset: DatasetBase,
        async_queue_size: int = 32,
    ) -> None:
        super().__init__(session, dataset)
        self.total_inference_time = 0.0  # Total time spent on inference
        self.recent_inference_times = deque(maxlen=200)  # total:50000
        self.async_queue_size = async_queue_size
        self._use_async = session.type == SessionType.dxruntime

    def eval(self) -> None:
        """evaluation IC Model."""
        if self._use_async:
            logger.info(f"Using async evaluation with queue_size={self.async_queue_size}")
            return self._eval_async()
        return self._eval_sync()

    def _eval_sync(self) -> dict:
        """Synchronous evaluation (original behavior)."""
        loader = self.make_loader()
        total_len = len(loader)

        topk_correct_count = [0, 0]
        current_count = 0

        pbar = tqdm(enumerate(loader), total=total_len)
        for batch, (image, label) in pbar:
            batch_size = image.size()[0]
            current_count += batch_size

            correct = self._run_one_batch(image, label)
            topk_correct_count = self._topk_eval(topk_correct_count, correct)

            if len(self.recent_inference_times) > 0:
                current_fps = len(self.recent_inference_times) / sum(self.recent_inference_times)
            else:
                current_fps = 0.0

            pbar.desc = (
                f"ImageNet | "
                f"Top1:{topk_correct_count[0]/current_count:.2f} "
                f"Top5:{topk_correct_count[1]/current_count:.2f} "
                f"Current_FPS:{current_fps:.1f}"
            )

        return self._finalize_results(topk_correct_count, current_count)

    def _eval_async(self) -> dict:
        """Asynchronous evaluation for NPU session using native async API.

        Uses separate threads for submission and result collection to maximize
        NPU utilization through true pipelining.

        Architecture:
            [Main Thread]                    [Worker Thread]
                 |                                 |
            run_async(img1) ──job_id──→      wait(job_id)
            run_async(img2)                  postprocessing
            run_async(img3)                  wait(job_id)
                ...                              ...
        """
        loader = self.make_loader()
        total_len = len(loader)

        # Shared state (thread-safe via queue and lock)
        job_queue: queue.Queue = queue.Queue(maxsize=self.async_queue_size)
        result_lock = threading.Lock()

        # Results collected by worker thread
        topk_correct_count = [0, 0]
        current_count = 0
        worker_done = threading.Event()

        # Wall-clock time measurement
        wall_start_time = time.time()

        pbar = tqdm(total=total_len)

        def worker_thread():
            """Worker thread: wait for jobs and process results."""
            nonlocal topk_correct_count, current_count

            while True:
                try:
                    item = job_queue.get(timeout=1.0)
                except queue.Empty:
                    if worker_done.is_set() and job_queue.empty():
                        break
                    continue

                if item is None:  # Poison pill
                    break

                job_id, label = item

                # Wait for NPU to complete this job
                output = self.session.wait(job_id)

                # Postprocessing
                batch_size = label.size()[0]

                output = self.postprocessing(output)
                label_np = torch_to_numpy(label)
                label_np = np.reshape(label_np, [-1, 1])
                correct = np.equal(output, label_np)

                with result_lock:
                    current_count += batch_size
                    topk_correct_count = self._topk_eval(topk_correct_count, correct)

                    # Update progress bar
                    elapsed = time.time() - wall_start_time
                    current_fps = current_count / elapsed if elapsed > 0 else 0.0

                    pbar.desc = (
                        f"ImageNet | "
                        f"Top1:{topk_correct_count[0]/current_count:.2f} "
                        f"Top5:{topk_correct_count[1]/current_count:.2f} "
                        f"Current_FPS:{current_fps:.1f}"
                    )
                    pbar.update(1)

                job_queue.task_done()

        # Start worker thread
        worker = threading.Thread(target=worker_thread, daemon=True)
        worker.start()

        # Main thread: submit async jobs
        for image, label in loader:
            job_id = self.session.run_async(image, user_arg=label)
            job_queue.put((job_id, label))  # Blocks if queue is full

        # Signal completion and wait for worker
        worker_done.set()
        job_queue.put(None)  # Poison pill
        worker.join()

        pbar.close()

        # Calculate total wall-clock time
        self.total_inference_time = time.time() - wall_start_time
        return self._finalize_results(topk_correct_count, current_count)

    def _finalize_results(self, topk_correct_count: List[int], current_count: int) -> dict:
        """Calculate and log final metrics."""
        top1_acc = (topk_correct_count[0] / current_count) * 100
        top5_acc = (topk_correct_count[1] / current_count) * 100
        avg_fps = current_count / self.total_inference_time if self.total_inference_time > 0 else 0

        # Print and log results
        print(f"Top1 Accuracy: {top1_acc:.2f}\n" f"Top5 Accuracy:  {top5_acc:.2f}\n" f"Average FPS: {avg_fps:.2f}")
        logger.success(f"@JSON <Top1 Accuracy:{top1_acc:.2f}; Top5 Accuracy:{top5_acc:.2f}; Average FPS:{avg_fps:.2f}>")

        return {
            "performance": [top1_acc, top5_acc],
            "fps": avg_fps,
        }

    def _run_one_batch(self, image: torch.Tensor, label: torch.Tensor) -> np.ndarray:
        """run one batch.

        Args:
            image (torch.Tensor): batch image.
            label (torch.Tensor): batch label.

        Returns:
            np.ndarray: run output.
        """
        start_time = time.time()
        output = self.session.run(image)
        inference_time = time.time() - start_time

        self.recent_inference_times.append(inference_time)
        self.total_inference_time += inference_time

        output = self.postprocessing(output)
        label = torch_to_numpy(label)
        label = np.reshape(label, [-1, 1])
        return np.equal(output, label)

    def _topk_eval(self, topk_correct_count: List[int], correct: np.ndarray, topk=[1, 5]) -> List[int]:
        """topk evaluation.

        Args:
            topk_correct_count (List[int]): topk correct count list.
            correct (np.ndarray): correct.
            topk (list, optional): topk value.. Defaults to [1, 5].

        Returns:
            List[int]: updated topk correct count list.
        """
        for idx_k, k in enumerate(topk):
            topk_correct_count[idx_k] += np.sum(correct[..., :k])
        return topk_correct_count
