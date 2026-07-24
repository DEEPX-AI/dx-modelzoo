from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.evaluator import EVALUATOR_REGISTRY, EvaluatorBase
from dx_modelzoo.session import SessionBase


@EVALUATOR_REGISTRY.register("image_classification")
class ImageClassificationEvaluator(EvaluatorBase):
    """Image Classification Evaluator (TopK accuracy)."""

    def __init__(self, session: SessionBase, dataset: DatasetBase, **kwargs) -> None:
        super().__init__(session, dataset, **kwargs)

    def init_metrics(self) -> dict:
        return {
            "topk_correct_count": [0, 0],
            "current_count": 0,
        }

    def extract_inputs(self, batch_data: Tuple) -> np.ndarray:
        image, label = batch_data
        if image.ndim == 3:
            image = np.expand_dims(image, 0)
        return image

    def process_batch_result(self, batch_data: Tuple, output: Any, metrics_state: dict) -> dict:
        image, label = batch_data
        label_np = np.array(label).reshape(-1, 1)
        correct = np.equal(output, label_np)
        metrics_state["current_count"] += 1
        metrics_state["topk_correct_count"] = self._topk_eval(metrics_state["topk_correct_count"], correct)
        return metrics_state

    def compute_final_metrics(self, metrics_state: dict) -> dict:
        tc = metrics_state["topk_correct_count"]
        n = metrics_state["current_count"]
        top1 = (tc[0] / n) * 100 if n > 0 else 0
        top5 = (tc[1] / n) * 100 if n > 0 else 0
        avg_fps = n / self.total_inference_time if self.total_inference_time > 0 else 0

        return self._finalize(
            metric_names=["Top1 Accuracy", "Top5 Accuracy"],
            metric_values=[top1, top5],
            fps=avg_fps,
        )

    def format_progress_desc(self, metrics_state: dict, current_fps: float) -> str:
        tc = metrics_state["topk_correct_count"]
        n = metrics_state["current_count"]
        if n == 0:
            return "ImageNet | Initializing..."
        top1 = tc[0] / n
        top5 = tc[1] / n
        return f"ImageNet | Top1:{top1:.2f} Top5:{top5:.2f} FPS:{current_fps:.1f}"

    def _topk_eval(
        self,
        topk_correct_count: List[int],
        correct: np.ndarray,
        topk: List[int] = None,
    ) -> List[int]:
        if topk is None:
            topk = [1, 5]
        for idx_k, k in enumerate(topk):
            topk_correct_count[idx_k] += int(np.sum(correct[..., :k]))
        return topk_correct_count
