from collections import deque

import numpy as np
import torch
from tqdm import tqdm

from dx_modelzoo.dataset import DatasetBase
from dx_modelzoo.evaluator import EvaluatorBase
from dx_modelzoo.session import SessionBase

from loguru import logger
import time 


class SegentationEvaluator(EvaluatorBase):
    """Segmentation Evaluator."""

    def __init__(self, session: SessionBase, dataset: DatasetBase):
        super().__init__(session, dataset)
        self.num_class = self.dataset.num_class
        self.total_inference_time = 0.0  
        self.recent_inference_times = deque(maxlen=15) # total : 1449

    def eval(self) -> None:
        loader = self.make_loader()
        total_len= len(loader)       
        confusion_matrix = np.zeros([self.num_class, self.num_class])
        pbar = tqdm(loader, total=total_len)
        for batch in pbar:
            confusion_matrix = self._run_one_batch(*batch, confusion_matrix)
            
            if len(self.recent_inference_times) > 0:
                current_fps = len(self.recent_inference_times) / sum(self.recent_inference_times)
            else:
                current_fps = 0.0
                
            pbar.desc = (
                f"Cityscapes | Current_FPS:{current_fps:.1f}"
            )

        # Calculate final metrics
        miou = self.calculate_miou(confusion_matrix)
        avg_fps = total_len / self.total_inference_time if self.total_inference_time > 0 else 0.0
        
        print(f"mIoU: {round(miou * 100, 3)}")
        print(f"Average FPS: {avg_fps:.2f}")
        logger.success(f"@JSON <mIoU:{round(miou * 100, 3)}; Average FPS:{avg_fps:.2f}>")

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
