from typing import Any

import numpy as np
import torch
from loguru import logger

from dx_modelzoo.evaluator import EvaluatorBase


class LFWEvaluator(EvaluatorBase):
    """LFW face-verification evaluator.

    The dataset yields ``(img1, img2, label)`` triplets.
    For every pair we run the model on *both* images, compute the cosine
    similarity of the resulting embeddings, and sweep a threshold over the
    6 000 pairs to report accuracy / best-threshold / AUC.
    """

    def __init__(self, session, dataset):
        super().__init__(session, dataset, workers=4)

    # ------------------------------------------------------------------
    # EvaluatorBase interface
    # ------------------------------------------------------------------

    def init_metrics(self) -> dict:
        return {
            "embeddings1": [],
            "embeddings2": [],
            "labels": [],
        }

    def extract_inputs(self, batch_data: Any) -> torch.Tensor:
        """Return first image; second image is handled in process_batch_result."""
        img1, _img2, _label = batch_data
        return img1

    def process_batch_result(
        self,
        batch_data: Any,
        output: Any,
        metrics_state: dict,
    ) -> dict:
        """Run second image through model, accumulate both embeddings."""
        _img1, img2, label = batch_data

        emb1 = self._to_numpy(output)

        # Run second image
        output2 = self.session.run(img2)
        if not getattr(self, "lazy_postprocessing", False):
            output2 = self.postprocessing(output2)
        emb2 = self._to_numpy(output2)

        metrics_state["embeddings1"].append(emb1)
        metrics_state["embeddings2"].append(emb2)
        metrics_state["labels"].append(int(label))

        return metrics_state

    def compute_final_metrics(self, metrics_state: dict) -> dict:
        emb1 = np.vstack(metrics_state["embeddings1"])
        emb2 = np.vstack(metrics_state["embeddings2"])
        labels = np.array(metrics_state["labels"])

        # L2-normalise
        emb1 = emb1 / (np.linalg.norm(emb1, axis=1, keepdims=True) + 1e-10)
        emb2 = emb2 / (np.linalg.norm(emb2, axis=1, keepdims=True) + 1e-10)

        # Cosine similarity
        cos_sim = np.sum(emb1 * emb2, axis=1)

        # Standard LFW 10-fold cross-validation
        n_folds = 10
        n_pairs = len(labels)
        fold_size = n_pairs // n_folds
        indices = np.arange(n_pairs)

        fold_accs = []
        fold_thrs = []
        for fold in range(n_folds):
            val_idx = indices[fold * fold_size : (fold + 1) * fold_size]
            train_idx = np.concatenate([indices[: fold * fold_size], indices[(fold + 1) * fold_size :]])

            # Find best threshold on train folds
            train_sim = cos_sim[train_idx]
            train_labels = labels[train_idx]
            best_thr, best_train_acc = 0.0, 0.0
            for thr in np.arange(-1.0, 1.01, 0.005):
                acc = np.mean((train_sim >= thr).astype(np.int32) == train_labels)
                if acc > best_train_acc:
                    best_train_acc, best_thr = acc, thr

            # Evaluate on val fold
            val_sim = cos_sim[val_idx]
            val_labels = labels[val_idx]
            val_acc = float(np.mean((val_sim >= best_thr).astype(np.int32) == val_labels))
            fold_accs.append(val_acc)
            fold_thrs.append(best_thr)

        mean_acc = float(np.mean(fold_accs))
        std_acc = float(np.std(fold_accs))
        mean_thr = float(np.mean(fold_thrs))
        avg_fps = n_pairs / self.total_inference_time if self.total_inference_time > 0 else 0.0

        print(f"LFW Accuracy: {mean_acc:.4f} ± {std_acc:.4f} (10-fold CV)")
        print(f"Mean Threshold: {mean_thr:.3f}")
        print(f"Average FPS: {avg_fps:.2f}")
        logger.success(
            f"@JSON <LFW Accuracy:{mean_acc:.4f}; Std:{std_acc:.4f}; "
            f"Threshold:{mean_thr:.3f}; Average FPS:{avg_fps:.2f}>"
        )

        return {
            "performance": [mean_acc],
            "fps": avg_fps,
        }

    def format_progress_desc(self, metrics_state: dict, current_fps: float) -> str:
        n = len(metrics_state["labels"])
        return f"LFW | Pairs:{n} Current_FPS:{current_fps:.1f}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_numpy(output) -> np.ndarray:
        if isinstance(output, (list, tuple)):
            output = output[0]
        if isinstance(output, torch.Tensor):
            output = output.detach().cpu().numpy()
        return np.atleast_2d(np.squeeze(output))
