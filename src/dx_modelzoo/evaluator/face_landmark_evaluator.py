import numpy as np
import torch
from loguru import logger

from dx_modelzoo.dataset.face_landmark import FaceLandmarkDataset
from dx_modelzoo.evaluator import EvaluatorBase
from dx_modelzoo.session import SessionBase

YAW_BINS = [(0, 30), (30, 60), (60, 90)]


class FaceLandmarkEvaluator(EvaluatorBase):
    """Face Landmark Evaluator using NME (Normalized Mean Error).

    Computes per-sample mean L2 error across 68 landmarks,
    normalized by face bounding box diagonal. Results are grouped
    by absolute yaw angle ranges following the 3DDFA evaluation protocol.

    Args:
        session: Runtime session.
        dataset: AFLW2000-3D face landmark dataset.
    """

    def __init__(self, session: SessionBase, dataset: FaceLandmarkDataset):
        super().__init__(session, dataset, workers=4)
        self.dataset: FaceLandmarkDataset

    def init_metrics(self) -> dict:
        return {
            "nme_per_bin": {f"{lo}_{hi}": [] for lo, hi in YAW_BINS},
            "all_nme": [],
            "count": 0,
        }

    def extract_inputs(self, batch_data) -> torch.Tensor:
        image, gt_landmarks, bbox_size, yaw_deg, idx = batch_data
        return image

    def process_batch_result(self, batch_data, output, metrics_state: dict) -> dict:
        image, gt_landmarks, bbox_size, yaw_deg, idx = batch_data

        pred_landmarks = output  # [68, 2] from postprocessing

        if isinstance(pred_landmarks, torch.Tensor):
            pred_landmarks = pred_landmarks.numpy()
        if isinstance(gt_landmarks, torch.Tensor):
            gt_landmarks = gt_landmarks.numpy()

        if pred_landmarks.ndim == 3:
            pred_landmarks = pred_landmarks[0]
        if gt_landmarks.ndim == 3:
            gt_landmarks = gt_landmarks[0]

        bbox_val = float(bbox_size.item() if hasattr(bbox_size, "item") else bbox_size)
        yaw_val = float(yaw_deg.item() if hasattr(yaw_deg, "item") else yaw_deg)

        # NME (%): mean L2 distance / bbox diagonal * 100
        distances = np.sqrt(np.sum((pred_landmarks - gt_landmarks) ** 2, axis=1))
        nme = np.mean(distances) / max(bbox_val, 1e-6) * 100

        metrics_state["all_nme"].append(nme)
        metrics_state["count"] += 1

        # Bin by yaw angle
        for lo, hi in YAW_BINS:
            if lo <= yaw_val < hi or (hi == 90 and yaw_val >= 60):
                metrics_state["nme_per_bin"][f"{lo}_{hi}"].append(nme)
                break

        return metrics_state

    def compute_final_metrics(self, metrics_state: dict) -> dict:
        count = metrics_state["count"]
        avg_fps = count / self.total_inference_time if self.total_inference_time > 0 else 0.0

        mean_nme = np.mean(metrics_state["all_nme"]) if metrics_state["all_nme"] else 0.0

        print(f"{'Yaw Range':<15} {'NME':<12} {'Count':<8}")
        print("-" * 35)
        for lo, hi in YAW_BINS:
            key = f"{lo}_{hi}"
            nmes = metrics_state["nme_per_bin"][key]
            bin_nme = np.mean(nmes) if nmes else 0.0
            print(f"[{lo}, {hi}]{'':>8} {bin_nme:.4f}{'':>6} {len(nmes)}")

        print(f"{'Mean':<15} {mean_nme:.4f}{'':>6} {count}")
        print(f"NME: {mean_nme:.4f}")
        print(f"Average Inference FPS: {avg_fps:.2f}")
        logger.success(f"@JSON <NME:{mean_nme:.6f}; Average FPS:{avg_fps:.2f}>")

        return {
            "performance": [mean_nme],
            "fps": avg_fps,
        }

    def format_progress_desc(self, metrics_state: dict, current_fps: float) -> str:
        count = metrics_state.get("count", 0)
        avg_nme = np.mean(metrics_state["all_nme"]) if metrics_state["all_nme"] else 0.0
        return f"FaceLandmark | NME:{avg_nme:.4f} Count:{count} FPS:{current_fps:.1f}"
