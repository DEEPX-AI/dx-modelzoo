from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.evaluator import EVALUATOR_REGISTRY, EvaluatorBase
from dx_modelzoo.session import SessionBase


@EVALUATOR_REGISTRY.register("semantic_segmentation")
@EVALUATOR_REGISTRY.register("person_segmentation")
class SegmentationEvaluator(EvaluatorBase):
    """Segmentation Evaluator (mIoU).

    The ``person_segmentation`` alias targets person-vs-background 2-class mIoU
    (e.g. NVIDIA PeopleSemSeg on ``COCOPersonSeg``); ``num_class`` is taken from
    the dataset, so no separate evaluator class is required.
    """

    def __init__(self, session: SessionBase, dataset: DatasetBase, **kwargs) -> None:
        super().__init__(session, dataset, workers=12, **kwargs)
        self.num_class = self.dataset.num_class

    def init_metrics(self) -> dict:
        return {"confusion_matrix": np.zeros([self.num_class, self.num_class])}

    def extract_inputs(self, batch_data: Tuple) -> np.ndarray:
        image, label = batch_data
        return image

    def process_batch_result(self, batch_data: Tuple, output: np.ndarray, metrics_state: dict) -> dict:
        image, label = batch_data
        confusion_matrix = metrics_state["confusion_matrix"]
        confusion_matrix = self._update_confusion_matrix(output, label, confusion_matrix)
        metrics_state["confusion_matrix"] = confusion_matrix
        return metrics_state

    def compute_final_metrics(self, metrics_state: dict) -> dict:
        confusion_matrix = metrics_state["confusion_matrix"]
        miou = self.calculate_miou(confusion_matrix)
        avg_fps = len(self.dataset) / self.total_inference_time if self.total_inference_time > 0 else 0.0
        return self._finalize(
            metric_names=["mIoU"],
            metric_values=[miou * 100],
            fps=avg_fps,
        )

    def format_progress_desc(self, metrics_state: dict, current_fps: float) -> str:
        return f"{self.dataset.__class__.__name__} | Current_FPS:{current_fps:.1f}"

    def _update_confusion_matrix(self, output: np.ndarray, label: Any, confusion_matrix: np.ndarray) -> np.ndarray:
        if not isinstance(label, np.ndarray):
            label = np.asarray(label)
        if not isinstance(output, np.ndarray):
            output = np.asarray(output)
        # Remove batch dimension if present
        while output.ndim > label.ndim:
            output = output[0]
        mask = (label >= 0) & (label < self.num_class)
        label_masked = self.num_class * label[mask].astype("int") + output[mask]
        bin_count = np.bincount(label_masked, minlength=self.num_class**2)
        confusion_matrix += bin_count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def calculate_miou(self, confusion_matrix: np.ndarray) -> float:
        denominator = (
            np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix)
        )
        valid = denominator > 0
        if not np.any(valid):
            return 0.0
        miou = np.diag(confusion_matrix)[valid] / denominator[valid]
        return float(np.mean(miou))
