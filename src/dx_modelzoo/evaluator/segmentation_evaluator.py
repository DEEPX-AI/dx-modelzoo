import queue
import threading
import time
from collections import deque

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from dx_modelzoo.dataset import DatasetBase
from dx_modelzoo.enums import SessionType
from dx_modelzoo.evaluator import EvaluatorBase
from dx_modelzoo.session import SessionBase


class SegentationEvaluator(EvaluatorBase):
    """Segmentation Evaluator."""

    def __init__(self, session: SessionBase, dataset: DatasetBase):
        super().__init__(session, dataset)
        self.num_class = self.dataset.num_class
        self.total_inference_time = 0.0
        self.recent_inference_times = deque(maxlen=15)  # total : 1449

    def eval(self) -> None:
        # Use async evaluation for DX runtime
        if self.session.type == SessionType.dxruntime:
            return self._eval_async()

        loader = self.make_loader()
        total_len = len(loader)
        confusion_matrix = np.zeros([self.num_class, self.num_class])
        pbar = tqdm(loader, total=total_len)
        for batch in pbar:
            confusion_matrix = self._run_one_batch(*batch, confusion_matrix)

            if len(self.recent_inference_times) > 0:
                current_fps = len(self.recent_inference_times) / sum(self.recent_inference_times)
            else:
                current_fps = 0.0
                
            pbar.desc = (
                f"{self.dataset.__class__.__name__} | Current_FPS:{current_fps:.1f}"
            )

        # Calculate final metrics
        miou = self.calculate_miou(confusion_matrix)
        avg_fps = total_len / self.total_inference_time if self.total_inference_time > 0 else 0.0

        print(f"mIoU: {round(miou * 100, 3)}")
        print(f"Average FPS: {avg_fps:.2f}")
        logger.success(f"@JSON <mIoU:{round(miou * 100, 3)}; Average FPS:{avg_fps:.2f}>")

        return {
            "performance": [miou * 100],
            "fps": avg_fps,
        }

    def _eval_async(self) -> dict:
        """Async evaluation for DX runtime using run_async/wait pattern."""
        loader = self.make_loader()
        total_len = len(loader)

        # Shared state with lock
        lock = threading.Lock()
        confusion_matrix = np.zeros([self.num_class, self.num_class])
        worker_error = None

        # Queue for pending jobs: (job_id, label)
        pending_queue = queue.Queue(maxsize=32)
        done_event = threading.Event()

        # Wall-clock time measurement
        wall_start_time = time.time()

        pbar = tqdm(total=total_len, desc="Cityscapes")

        def result_worker():
            """Worker thread that collects results and updates confusion matrix."""
            nonlocal confusion_matrix, worker_error

            while True:
                try:
                    item = pending_queue.get(timeout=0.1)
                except queue.Empty:
                    if done_event.is_set():
                        break
                    continue

                if item is None:  # Sentinel
                    break

                job_id, label = item

                try:
                    # Wait for NPU result
                    output = self.session.wait(job_id)

                    # Postprocessing
                    output = self.postprocessing(output)

                    with lock:
                        # Update confusion matrix
                        confusion_matrix = self._update_confusion_matrix(output, label, confusion_matrix)

                        # Wall-clock based FPS
                        elapsed = time.time() - wall_start_time
                        processed_count = pbar.n + 1
                        current_fps = processed_count / elapsed if elapsed > 0 else 0.0

                        pbar.set_description(f"Cityscapes | Current_FPS:{current_fps:.1f}")
                        pbar.update(1)

                except Exception as e:
                    with lock:
                        worker_error = e
                    break

        # Start worker thread
        worker_thread = threading.Thread(target=result_worker, daemon=True)
        worker_thread.start()

        try:
            # Main thread: submit jobs
            for batch in loader:
                with lock:
                    if worker_error is not None:
                        raise worker_error

                image, label = batch
                job_id = self.session.run_async(image)

                pending_queue.put((job_id, label))

            # Signal completion and wait for worker
            done_event.set()
            pending_queue.put(None)  # Sentinel
            worker_thread.join()

            if worker_error is not None:
                raise worker_error

        finally:
            pbar.close()

        # Calculate final metrics
        total_wall_time = time.time() - wall_start_time
        miou = self.calculate_miou(confusion_matrix)
        avg_fps = total_len / total_wall_time if total_wall_time > 0 else 0.0

        print(f"mIoU: {round(miou * 100, 3)}")
        print(f"Average FPS: {avg_fps:.2f}")
        logger.success(f"@JSON <mIoU:{round(miou * 100, 3)}; Average FPS:{avg_fps:.2f}>")

        return {
            "performance": [miou * 100],
            "fps": avg_fps,
        }

    def _run_one_batch(self, image: torch.Tensor, label: torch.Tensor, confusion_matrix: np.ndarray):
        start_time = time.time()
        output = self.session.run(image)
        inference_time = time.time() - start_time

        self.recent_inference_times.append(inference_time)
        self.total_inference_time += inference_time
        output = self.postprocessing(output)
        confusion_matrix = self._update_confusion_matrix(output, label, confusion_matrix)
        return confusion_matrix

    def _update_confusion_matrix(
        self, output: np.ndarray, label: torch.Tensor, confusion_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Updates the confusion matrix based on the model's output and the true labels.

        This function computes the confusion matrix for a multi-class segmentation task by comparing
        the predicted output and ground truth labels, and updates the given confusion matrix in-place.

        Args:
            output (np.ndarray): The predicted output from the model, a 1D array of predicted class indices.
            label (torch.Tensor): The ground truth labels, a tensor of true class indices.
            confusion_matrix (np.ndarray): The current confusion matrix to be updated, initially a square matrix of
                                            zeros with shape (num_class, num_class), where num_class is the number of
                                            classes.

        Returns:
            np.ndarray: The updated confusion matrix after incorporating the new batch of predictions.

        Notes:
            The function assumes that the `output` and `label` arrays are of the same shape and that the classes are
            indexed starting from 0 to num_class - 1. The mask filters out invalid or ignored labels (e.g., labels
            outside valid class range).

        Example:
            >>> confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
            >>> confusion_matrix = _update_confusion_matrix(output, label, confusion_matrix)
        """
        label = label.numpy()
        mask = (label >= 0) & (label < self.num_class)

        label = self.num_class * label[mask].astype("int") + output[mask]
        bin_count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix += bin_count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def calculate_miou(self, confusion_matrix):
        """
        Calculates the mean Intersection over Union (mIoU) from a given confusion matrix.

        The mIoU is a common evaluation metric for semantic segmentation tasks, computed as
        the average of IoU (Intersection over Union) for each class. The IoU for each class
        is calculated as the ratio of the intersection of predicted and true pixels to the
        union of predicted and true pixels.

        Args:
            confusion_matrix (np.ndarray): A square confusion matrix of shape (num_classes, num_classes)
                                        representing the counts of true vs predicted class labels.

        Returns:
            float: The mean Intersection over Union (mIoU) score, averaged across all classes.

        Notes:
            The function handles the case of undefined IoU (e.g., divisions by zero) by returning `NaN`
            for classes where the union of predicted and true pixels is zero, and it calculates the mean
            while ignoring those NaN values.

        Example:
            >>> miou_score = calculate_miou(confusion_matrix)
            >>> print(miou_score)  # e.g., 0.85
        """
        miou = np.diag(confusion_matrix) / (
            np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix)
        )
        return np.nanmean(miou)
