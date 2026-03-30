import numpy as np
import torch
from loguru import logger

from dx_modelzoo.dataset.peta import NUM_ATTRIBUTES, PETA_ATTRIBUTES
from dx_modelzoo.evaluator import EvaluatorBase
from dx_modelzoo.session import SessionBase


class PersonAttributeEvaluator(EvaluatorBase):
    """Person Attribute Recognition Evaluator using Average Accuracy.

    For each sample, computes per-attribute binary prediction (threshold=0.5).
    If model outputs raw logits, sigmoid is applied first; if outputs are
    already probabilities (all values in [0, 1]), sigmoid is skipped.
    Reports average accuracy across 35 selected attributes.

    Args:
        session (SessionBase): runtime session.
        dataset: PETA dataset.
    """

    def __init__(self, session: SessionBase, dataset):
        super().__init__(session, dataset, workers=4)

    def init_metrics(self) -> dict:
        return {
            "tp": np.zeros(NUM_ATTRIBUTES, dtype=np.int64),
            "fn": np.zeros(NUM_ATTRIBUTES, dtype=np.int64),
            "tn": np.zeros(NUM_ATTRIBUTES, dtype=np.int64),
            "fp": np.zeros(NUM_ATTRIBUTES, dtype=np.int64),
            "count": 0,
        }

    def extract_inputs(self, batch_data) -> torch.Tensor:
        image, labels, idx = batch_data
        return image

    def process_batch_result(self, batch_data, output, metrics_state: dict) -> dict:
        image, labels, idx = batch_data

        logits = output
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        if isinstance(logits, torch.Tensor):
            logits = logits.numpy()
        logits = np.squeeze(logits)  # [35]

        # Auto-detect: skip sigmoid if outputs are already probabilities
        if np.all((logits >= 0) & (logits <= 1)):
            probs = logits
        else:
            probs = 1.0 / (1.0 + np.exp(-logits))
        preds = (probs >= 0.5).astype(np.int64)

        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        labels = np.squeeze(labels)

        metrics_state["tp"] += ((preds == 1) & (labels == 1)).astype(np.int64)
        metrics_state["fn"] += ((preds == 0) & (labels == 1)).astype(np.int64)
        metrics_state["tn"] += ((preds == 0) & (labels == 0)).astype(np.int64)
        metrics_state["fp"] += ((preds == 1) & (labels == 0)).astype(np.int64)
        metrics_state["count"] += 1

        return metrics_state

    def compute_final_metrics(self, metrics_state: dict) -> dict:
        tp = metrics_state["tp"]
        fn = metrics_state["fn"]
        tn = metrics_state["tn"]
        fp = metrics_state["fp"]
        count = metrics_state["count"]

        # mA = mean of per-attribute accuracy: (TP/(TP+FN) + TN/(TN+FP)) / 2
        pos_acc = tp / np.maximum(tp + fn, 1)
        neg_acc = tn / np.maximum(tn + fp, 1)
        per_attr_acc = (pos_acc + neg_acc) / 2.0
        ma = float(np.mean(per_attr_acc))

        avg_fps = count / self.total_inference_time if self.total_inference_time > 0 else 0.0

        for name, acc in zip(PETA_ATTRIBUTES, per_attr_acc):
            print(f"  {name:30s}: {acc:.4f}")

        print(f"Average Accuracy: {ma:.6f}")
        print(f"Average Inference FPS: {avg_fps:.2f}")

        logger.success(f"@JSON <Average Accuracy:{ma:.6f}; Average FPS:{avg_fps:.2f}>")

        return {
            "performance": [ma],
            "fps": avg_fps,
        }

    def format_progress_desc(self, metrics_state: dict, current_fps: float) -> str:
        count = metrics_state.get("count", 0)
        if count > 0:
            tp = metrics_state["tp"]
            fn = metrics_state["fn"]
            tn = metrics_state["tn"]
            fp = metrics_state["fp"]
            pos_acc = tp / np.maximum(tp + fn, 1)
            neg_acc = tn / np.maximum(tn + fp, 1)
            ma = float(np.mean((pos_acc + neg_acc) / 2.0))
        else:
            ma = 0.0
        return f"PAR | AvgAcc:{ma:.6f} Count:{count} FPS:{current_fps:.1f}"
