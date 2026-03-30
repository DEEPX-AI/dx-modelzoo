import numpy as np
import torch
from loguru import logger

from dx_modelzoo.dataset.celeba import CELEBA_ATTRIBUTES, CelebADataset
from dx_modelzoo.evaluator import EvaluatorBase
from dx_modelzoo.session import SessionBase


class FaceAttributeEvaluator(EvaluatorBase):
    """Face Attribute Evaluator using Mean Attribute Accuracy.

    For each sample, computes per-attribute binary classification accuracy.
    Reports mean accuracy across all 40 attributes.

    Args:
        session (SessionBase): runtime session.
        dataset (CelebADataset): CelebA dataset.
    """

    def __init__(self, session: SessionBase, dataset: CelebADataset):
        super().__init__(session, dataset, workers=4)
        self.dataset: CelebADataset

    def init_metrics(self) -> dict:
        n = CelebADataset.NUM_ATTRIBUTES
        return {"correct": np.zeros(n, dtype=np.int64), "count": 0}

    def extract_inputs(self, batch_data) -> torch.Tensor:
        image, labels, idx = batch_data
        return image

    def process_batch_result(self, batch_data, output, metrics_state: dict) -> dict:
        image, labels, idx = batch_data

        # output[0]: [1, 40, 2] logits
        logits = output[0]
        if isinstance(logits, torch.Tensor):
            logits = logits.numpy()

        if logits.ndim == 3:
            logits = logits[0]  # [40, 2]

        # Predicted class: argmax over the 2 logits per attribute
        preds = np.argmax(logits, axis=1)  # [40]

        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        if labels.ndim == 2:
            labels = labels[0]

        metrics_state["correct"] += (preds == labels).astype(np.int64)
        metrics_state["count"] += 1

        return metrics_state

    def compute_final_metrics(self, metrics_state: dict) -> dict:
        count = metrics_state["count"]
        per_attr_acc = metrics_state["correct"] / max(count, 1)
        mean_acc = float(np.mean(per_attr_acc))

        avg_fps = count / self.total_inference_time if self.total_inference_time > 0 else 0.0

        # Per-attribute breakdown
        for name, acc in zip(CELEBA_ATTRIBUTES, per_attr_acc):
            print(f"  {name:30s}: {acc:.4f}")

        print(f"Average Accuracy: {mean_acc:.6f}")
        print(f"Average Inference FPS: {avg_fps:.2f}")

        logger.success(f"@JSON <AverageAccuracy:{mean_acc:.6f}; Average FPS:{avg_fps:.2f}>")

        return {
            "performance": [mean_acc],
            "fps": avg_fps,
        }

    def format_progress_desc(self, metrics_state: dict, current_fps: float) -> str:
        count = metrics_state.get("count", 0)
        if count > 0:
            mean_acc = float(np.mean(metrics_state["correct"] / count))
        else:
            mean_acc = 0.0
        return f"FaceAttr | AvgAcc:{mean_acc:.6f} Count:{count} FPS:{current_fps:.1f}"
