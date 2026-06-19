from __future__ import annotations

from typing import Any

import numpy as np

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.evaluator import EVALUATOR_REGISTRY, EvaluatorBase
from dx_modelzoo.session import SessionBase


@EVALUATOR_REGISTRY.register("face_recognition")
class FaceRecognitionEvaluator(EvaluatorBase):
    """LFW face-verification evaluator (accuracy)."""

    def __init__(self, session: SessionBase, dataset: DatasetBase, **kwargs) -> None:
        super().__init__(session, dataset, workers=12, **kwargs)

    def init_metrics(self) -> dict:
        return {"embeddings1": [], "embeddings2": [], "labels": []}

    def extract_inputs(self, batch_data: Any) -> np.ndarray:
        img1, _img2, _label = batch_data
        return img1

    def _run_postprocessing(self, output, batch_data):
        """Run second image through session and return both embeddings."""
        import time

        _img1, img2, label = batch_data
        emb1 = self._to_numpy(output)
        t0 = time.time()
        output2 = self.session.run(img2)
        self.total_inference_time += time.time() - t0
        if self.postprocessing is not None:
            output2 = self.postprocessing(output2)
        emb2 = self._to_numpy(output2)
        return (emb1, emb2)

    def process_batch_result(self, batch_data: Any, output: Any, metrics_state: dict) -> dict:
        _img1, img2, label = batch_data
        emb1, emb2 = output
        metrics_state["embeddings1"].append(emb1)
        metrics_state["embeddings2"].append(emb2)
        metrics_state["labels"].append(int(label))
        return metrics_state

    def compute_final_metrics(self, metrics_state: dict) -> dict:
        emb1 = np.vstack(metrics_state["embeddings1"])
        emb2 = np.vstack(metrics_state["embeddings2"])
        labels = np.array(metrics_state["labels"])
        emb1 = emb1 / (np.linalg.norm(emb1, axis=1, keepdims=True) + 1e-10)
        emb2 = emb2 / (np.linalg.norm(emb2, axis=1, keepdims=True) + 1e-10)
        cos_sim = np.sum(emb1 * emb2, axis=1)
        n_folds = 10
        n_pairs = len(labels)
        fold_size = n_pairs // n_folds
        indices = np.arange(n_pairs)
        fold_accs = []
        for fold in range(n_folds):
            val_idx = indices[fold * fold_size : (fold + 1) * fold_size]
            train_idx = np.concatenate([indices[: fold * fold_size], indices[(fold + 1) * fold_size :]])
            if len(val_idx) == 0 or len(train_idx) == 0:
                continue
            train_sim = cos_sim[train_idx]
            train_labels = labels[train_idx]
            best_thr, best_train_acc = 0.0, 0.0
            for thr in np.arange(-1.0, 1.01, 0.005):
                acc = np.mean((train_sim >= thr).astype(np.int32) == train_labels)
                if acc > best_train_acc:
                    best_train_acc, best_thr = acc, thr
            val_sim = cos_sim[val_idx]
            val_labels = labels[val_idx]
            val_acc = float(np.mean((val_sim >= best_thr).astype(np.int32) == val_labels))
            fold_accs.append(val_acc)
        mean_acc = float(np.mean(fold_accs)) if fold_accs else 0.0
        _std_acc = float(np.std(fold_accs)) if fold_accs else 0.0  # noqa: F841
        avg_fps = n_pairs / self.total_inference_time if self.total_inference_time > 0 else 0.0
        return self._finalize(
            metric_names=["LFW Accuracy"],
            metric_values=[mean_acc * 100],
            fps=avg_fps,
        )

    def format_progress_desc(self, metrics_state: dict, current_fps: float) -> str:
        n = len(metrics_state["labels"])
        return f"LFW | Pairs:{n} Current_FPS:{current_fps:.1f}"

    @staticmethod
    def _to_numpy(output) -> np.ndarray:
        if isinstance(output, (list, tuple)):
            output = output[0]
        return np.atleast_2d(np.squeeze(np.asarray(output)))
