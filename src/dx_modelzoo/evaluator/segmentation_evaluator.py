from typing import Tuple

import numpy as np
import torch
from loguru import logger

from dx_modelzoo.dataset import DatasetBase
from dx_modelzoo.evaluator import EvaluatorBase
from dx_modelzoo.session import SessionBase


class SegentationEvaluator(EvaluatorBase):
    """Segmentation Evaluator."""

    def __init__(self, session: SessionBase, dataset: DatasetBase):
        super().__init__(session, dataset, workers=12)
        self.num_class = self.dataset.num_class

    def init_metrics(self) -> dict:
        """Initialize metrics state."""
        return {
            "confusion_matrix": np.zeros([self.num_class, self.num_class]),
        }

    def extract_inputs(self, batch_data: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Extract image from batch data."""
        image, label = batch_data
        return image

    def process_batch_result(
        self,
        batch_data: Tuple[torch.Tensor, torch.Tensor],
        output: np.ndarray,
        metrics_state: dict,
    ) -> dict:
        """Process batch result and update confusion matrix."""
        image, label = batch_data

        confusion_matrix = metrics_state["confusion_matrix"]
        confusion_matrix = self._update_confusion_matrix(output, label, confusion_matrix)
        metrics_state["confusion_matrix"] = confusion_matrix

        return metrics_state

    def compute_final_metrics(self, metrics_state: dict) -> dict:
        """Compute final mIoU metric."""
        confusion_matrix = metrics_state["confusion_matrix"]
        miou = self.calculate_miou(confusion_matrix)
        avg_fps = len(self.dataset) / self.total_inference_time if self.total_inference_time > 0 else 0.0

        print(f"mIoU: {round(miou * 100, 3)}")
        print(f"Average FPS: {avg_fps:.2f}")
        logger.success(f"@JSON <mIoU:{round(miou * 100, 3)}; Average FPS:{avg_fps:.2f}>")

        return {
            "performance": [miou * 100],
            "fps": avg_fps,
        }

    def format_progress_desc(self, metrics_state: dict, current_fps: float) -> str:
        """Format progress bar description."""
        return f"{self.dataset.__class__.__name__} | Current_FPS:{current_fps:.1f}"

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
