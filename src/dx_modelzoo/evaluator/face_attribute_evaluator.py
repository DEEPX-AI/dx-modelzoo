from __future__ import annotations

from typing import Any

import numpy as np

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.evaluator import EVALUATOR_REGISTRY, EvaluatorBase
from dx_modelzoo.session import SessionBase


@EVALUATOR_REGISTRY.register("face_attribute")
class FaceAttributeEvaluator(EvaluatorBase):
    """Face Attribute Evaluator (average accuracy across 40 attributes)."""

    NUM_ATTRIBUTES = 40

    def __init__(self, session: SessionBase, dataset: DatasetBase, **kwargs) -> None:
        super().__init__(session, dataset, workers=12, **kwargs)

    def init_metrics(self) -> dict:
        return {"correct": np.zeros(self.NUM_ATTRIBUTES, dtype=np.int64), "count": 0}

    def extract_inputs(self, batch_data: Any) -> np.ndarray:
        image, labels, idx = batch_data
        return image

    def process_batch_result(self, batch_data: Any, output: Any, metrics_state: dict) -> dict:
        image, labels, idx = batch_data
        logits = output[0] if isinstance(output, (list, tuple)) else output
        logits = np.asarray(logits)
        if logits.ndim == 3:
            logits = logits[0]
        preds = np.argmax(logits, axis=1)
        labels = np.asarray(labels)
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
        return self._finalize(
            metric_names=["Average Accuracy"],
            metric_values=[mean_acc * 100],
            fps=avg_fps,
        )

    def format_progress_desc(self, metrics_state: dict, current_fps: float) -> str:
        count = metrics_state.get("count", 0)
        mean_acc = float(np.mean(metrics_state["correct"] / count)) if count > 0 else 0.0
        return f"FaceAttr | AvgAcc:{mean_acc:.6f} Count:{count} FPS:{current_fps:.1f}"
