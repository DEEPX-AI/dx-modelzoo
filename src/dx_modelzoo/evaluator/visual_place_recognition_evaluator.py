from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.evaluator import EVALUATOR_REGISTRY, EvaluatorBase
from dx_modelzoo.session import SessionBase



@EVALUATOR_REGISTRY.register("visual_place_recognition")
class VisualPlaceRecognitionEvaluator(EvaluatorBase):
    """Visual place recognition evaluator (HPatches surrogate).

    Within each HPatches sequence the first image (``n=1``) acts as the
    query reference and the remaining 5 images (``n=2..6``) as the
    gallery.  Recall@1 across all sequences is reported by checking
    whether the gallery descriptor closest to the reference belongs to
    the same sequence.

    Requires the dataset to expose ``(descriptor_input, shape, seq, n,
    path)`` as the dataset tuple (see ``HPatches``).
    """

    def __init__(self, session: SessionBase, dataset: DatasetBase, **kwargs) -> None:
        super().__init__(session, dataset, **kwargs)

    def init_metrics(self) -> dict:
        return {"refs": {}, "gallery": [], "current_count": 0}

    def extract_inputs(self, batch_data: Tuple) -> np.ndarray:
        image = batch_data[0]
        if isinstance(image, np.ndarray) and image.ndim == 3:
            image = np.expand_dims(image, 0)
        return image

    def process_batch_result(self, batch_data: Tuple, output: Any, metrics_state: dict) -> dict:
        descriptor = np.asarray(output, dtype=np.float32).ravel()
        seq = batch_data[2] if len(batch_data) > 2 else "_"
        n = int(batch_data[3]) if len(batch_data) > 3 else 1
        norm = float(np.linalg.norm(descriptor) + 1e-12)
        descriptor = descriptor / norm
        if n == 1:
            metrics_state["refs"][seq] = descriptor
        else:
            metrics_state["gallery"].append((seq, descriptor))
        metrics_state["current_count"] += 1
        return metrics_state

    def compute_final_metrics(self, metrics_state: dict) -> dict:
        refs = metrics_state["refs"]
        gallery: List[Tuple[str, np.ndarray]] = metrics_state["gallery"]
        n = metrics_state["current_count"]
        recall1 = 0.0
        if refs and gallery:
            ref_seqs = list(refs.keys())
            ref_mat = np.stack([refs[s] for s in ref_seqs], axis=0)
            correct = 0
            total = 0
            for seq, desc in gallery:
                if seq not in refs:
                    continue
                sims = ref_mat @ desc
                best = ref_seqs[int(np.argmax(sims))]
                correct += int(best == seq)
                total += 1
            recall1 = (correct / total) * 100 if total else 0.0

        avg_fps = n / self.total_inference_time if self.total_inference_time > 0 else 0
        return self._finalize(metric_names=["Recall@1"], metric_values=[recall1], fps=avg_fps)

    def format_progress_desc(self, metrics_state: dict, current_fps: float) -> str:
        n = metrics_state["current_count"]
        if n == 0:
            return "HPatches | Initializing..."
        return f"HPatches | N:{n} FPS:{current_fps:.1f}"
