# ic_evaluator.py
from typing import Any, List, Tuple

import numpy as np
import torch
from loguru import logger

from dx_modelzoo.dataset import DatasetBase
from dx_modelzoo.evaluator import EvaluatorBase
from dx_modelzoo.session import SessionBase
from dx_modelzoo.utils import torch_to_numpy


class ICEvaluator(EvaluatorBase):
    """Image Classification Evaluator."""

    def __init__(
        self,
        session: SessionBase,
        dataset: DatasetBase,
    ) -> None:
        super().__init__(session, dataset)

    def init_metrics(self) -> dict:
        """Initialize metrics state."""
        return {
            "topk_correct_count": [0, 0],
            "current_count": 0,
        }

    def extract_inputs(self, batch_data: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Extract image from batch data."""
        image, label = batch_data
        return image

    def process_batch_result(
        self,
        batch_data: Tuple[torch.Tensor, torch.Tensor],
        output: Any,
        metrics_state: dict,
    ) -> dict:
        """Process batch result and update top-k accuracy."""
        image, label = batch_data
        batch_size = label.size()[0]

        # Convert to numpy and reshape
        label_np = torch_to_numpy(label)
        label_np = np.reshape(label_np, [-1, 1])
        correct = np.equal(output, label_np)

        # Update metrics
        metrics_state["current_count"] += batch_size
        metrics_state["topk_correct_count"] = self._topk_eval(metrics_state["topk_correct_count"], correct)

        return metrics_state

    def compute_final_metrics(self, metrics_state: dict) -> dict:
        """Compute final top-1 and top-5 accuracy."""
        topk_correct_count = metrics_state["topk_correct_count"]
        current_count = metrics_state["current_count"]

        top1_acc = (topk_correct_count[0] / current_count) * 100
        top5_acc = (topk_correct_count[1] / current_count) * 100
        avg_fps = current_count / self.total_inference_time if self.total_inference_time > 0 else 0

        # Log results
        print(f"Top1 Accuracy: {top1_acc:.2f}\nTop5 Accuracy: {top5_acc:.2f}\nAverage FPS: {avg_fps:.2f}")
        logger.success(f"@JSON <Top1 Accuracy:{top1_acc:.2f}; Top5 Accuracy:{top5_acc:.2f}; Average FPS:{avg_fps:.2f}>")

        return {
            "performance": [top1_acc, top5_acc],
            "fps": avg_fps,
        }

    def format_progress_desc(self, metrics_state: dict, current_fps: float) -> str:
        """Format progress bar description."""
        topk_correct_count = metrics_state["topk_correct_count"]
        current_count = metrics_state["current_count"]

        if current_count == 0:
            return "ImageNet | Initializing..."

        top1_acc = topk_correct_count[0] / current_count
        top5_acc = topk_correct_count[1] / current_count

        return f"ImageNet | " f"Top1:{top1_acc:.2f} " f"Top5:{top5_acc:.2f} " f"Current_FPS:{current_fps:.1f}"

    def _topk_eval(self, topk_correct_count: List[int], correct: np.ndarray, topk=[1, 5]) -> List[int]:
        """Update top-k correct count."""
        for idx_k, k in enumerate(topk):
            topk_correct_count[idx_k] += np.sum(correct[..., :k])
        return topk_correct_count
