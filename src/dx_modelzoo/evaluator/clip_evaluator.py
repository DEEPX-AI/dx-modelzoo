from typing import Tuple

import numpy as np
import torch
from loguru import logger

from dx_modelzoo.dataset import DatasetBase
from dx_modelzoo.evaluator import EvaluatorBase
from dx_modelzoo.session import SessionBase


class CLIPEvaluator(EvaluatorBase):
    def __init__(
        self,
        session: SessionBase,
        dataset: DatasetBase,
        zero_shot_text_embedding: str,
    ) -> None:
        super().__init__(session, dataset, workers=12)
        self.zero_shot_text_embedding = zero_shot_text_embedding
        self.zeroshot_text_embedding_weight = None

    def _accuracy(self, output, target, topk=(1, 5)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        pred = output.topk(max(topk), 1, True, True)[1].t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [float(correct[:k].reshape(-1).float().sum(0, keepdims=True).cpu().numpy()) for k in topk]

    def init_metrics(self) -> dict:
        """Initialize metrics state."""
        # Load text embedding weight once
        self.zeroshot_text_embedding_weight = torch.from_numpy(np.load(self.zero_shot_text_embedding))

        return {
            "correct_top1": 0.0,
            "correct_top5": 0.0,
            "current_count": 0,
        }

    def extract_inputs(self, batch_data: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Extract images from batch data."""
        images, labels = batch_data
        return images

    def process_batch_result(
        self,
        batch_data: Tuple[torch.Tensor, torch.Tensor],
        output: np.ndarray,
        metrics_state: dict,
    ) -> dict:
        """Process batch result and update top-1 and top-5 accuracy."""
        images, labels = batch_data

        image_feature = torch.from_numpy(output)
        image_feature /= image_feature.norm(dim=-1, keepdim=True)
        logits = 100.0 * image_feature @ self.zeroshot_text_embedding_weight

        acc1, acc5 = self._accuracy(logits, labels, topk=(1, 5))

        batch_size = images.shape[0]
        metrics_state["correct_top1"] += acc1
        metrics_state["correct_top5"] += acc5
        metrics_state["current_count"] += batch_size

        return metrics_state

    def compute_final_metrics(self, metrics_state: dict) -> dict:
        """Compute final top-1 and top-5 accuracy."""
        correct_top1 = metrics_state["correct_top1"]
        correct_top5 = metrics_state["correct_top5"]
        current_count = metrics_state["current_count"]

        top1_acc = (correct_top1 / current_count * 100) if current_count > 0 else 0.0
        top5_acc = (correct_top5 / current_count * 100) if current_count > 0 else 0.0
        avg_fps = current_count / self.total_inference_time if self.total_inference_time > 0 else 0.0

        print(f"Top1 Accuracy: {top1_acc:.2f}")
        print(f"Top5 Accuracy: {top5_acc:.2f}")
        print(f"Average FPS: {avg_fps:.2f}")
        logger.success(
            f"@JSON <Top1 Accuracy:{top1_acc:.2f}; " f"Top5 Accuracy:{top5_acc:.2f}; " f"Average FPS:{avg_fps:.2f}>"
        )

        return {
            "performance": [correct_top1, correct_top5],
            "fps": avg_fps,
        }

    def format_progress_desc(self, metrics_state: dict, current_fps: float) -> str:
        """Format progress bar description."""
        correct_top1 = metrics_state["correct_top1"]
        correct_top5 = metrics_state["correct_top5"]
        current_count = metrics_state["current_count"]

        if current_count == 0:
            return "ImageNet | Initializing..."

        top1_ratio = correct_top1 / current_count
        top5_ratio = correct_top5 / current_count

        return f"ImageNet | " f"Top1:{top1_ratio:.2f} " f"Top5:{top5_ratio:.2f} " f"Current_FPS:{current_fps:.1f}"
