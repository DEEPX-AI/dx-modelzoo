from __future__ import annotations

from typing import Any

import numpy as np

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.evaluator import EVALUATOR_REGISTRY, EvaluatorBase
from dx_modelzoo.session import SessionBase

NUM_ATTRIBUTES = 35


@EVALUATOR_REGISTRY.register("person_attribute")
class PersonAttributeEvaluator(EvaluatorBase):
    """Person Attribute Recognition Evaluator (mA)."""

    def __init__(self, session: SessionBase, dataset: DatasetBase, **kwargs) -> None:
        super().__init__(session, dataset, workers=12, **kwargs)

    def init_metrics(self) -> dict:
        return {
            "tp": np.zeros(NUM_ATTRIBUTES, dtype=np.int64),
            "fn": np.zeros(NUM_ATTRIBUTES, dtype=np.int64),
            "tn": np.zeros(NUM_ATTRIBUTES, dtype=np.int64),
            "fp": np.zeros(NUM_ATTRIBUTES, dtype=np.int64),
            "count": 0,
        }

    def extract_inputs(self, batch_data: Any) -> np.ndarray:
        image, labels, idx = batch_data
        return image

    def process_batch_result(self, batch_data: Any, output: Any, metrics_state: dict) -> dict:
        image, labels, idx = batch_data
        logits = output[0] if isinstance(output, (list, tuple)) else output
        logits = np.squeeze(np.asarray(logits))
        if np.all((logits >= 0) & (logits <= 1)):
            probs = logits
        else:
            probs = 1.0 / (1.0 + np.exp(-logits))
        preds = (probs >= 0.5).astype(np.int64)
        labels = np.squeeze(np.asarray(labels))
        metrics_state["tp"] += ((preds == 1) & (labels == 1)).astype(np.int64)
        metrics_state["fn"] += ((preds == 0) & (labels == 1)).astype(np.int64)
        metrics_state["tn"] += ((preds == 0) & (labels == 0)).astype(np.int64)
        metrics_state["fp"] += ((preds == 1) & (labels == 0)).astype(np.int64)
        metrics_state["count"] += 1
        return metrics_state

    def compute_final_metrics(self, metrics_state: dict) -> dict:
        tp, fn = metrics_state["tp"], metrics_state["fn"]
        tn, fp = metrics_state["tn"], metrics_state["fp"]
        count = metrics_state["count"]
        pos_acc = tp / np.maximum(tp + fn, 1)
        neg_acc = tn / np.maximum(tn + fp, 1)
        per_attr_acc = (pos_acc + neg_acc) / 2.0
        ma = float(np.mean(per_attr_acc))
        avg_fps = count / self.total_inference_time if self.total_inference_time > 0 else 0.0
        return self._finalize(
            metric_names=["Average Accuracy"],
            metric_values=[ma * 100],
            fps=avg_fps,
        )

    def format_progress_desc(self, metrics_state: dict, current_fps: float) -> str:
        count = metrics_state.get("count", 0)
        if count > 0:
            tp, fn = metrics_state["tp"], metrics_state["fn"]
            tn, fp = metrics_state["tn"], metrics_state["fp"]
            ma = float(np.mean((tp / np.maximum(tp + fn, 1) + tn / np.maximum(tn + fp, 1)) / 2.0))
        else:
            ma = 0.0
        return f"PAR | AvgAcc:{ma:.6f} Count:{count} FPS:{current_fps:.1f}"
