import numpy as np
import torch
from loguru import logger

from dx_modelzoo.dataset.hand_landmark import HandLandmarkDataset
from dx_modelzoo.evaluator import EvaluatorBase
from dx_modelzoo.session import SessionBase

# MediaPipe hand landmark indices
WRIST = 0
MIDDLE_FINGER_MCP = 9


class HandLandmarkEvaluator(EvaluatorBase):
    """Hand Landmark Evaluator using MNAE (Mean of Normalized Absolute Error by palm size).

    For each hand sample, computes per-keypoint L2 error normalized by the palm size
    (Euclidean distance from WRIST to MIDDLE_FINGER_MCP in GT keypoints).

    Args:
        session (SessionBase): runtime session.
        dataset (HandLandmarkDataset): hand landmark dataset.
    """

    def __init__(self, session: SessionBase, dataset: HandLandmarkDataset):
        super().__init__(session, dataset, workers=4)
        self.dataset: HandLandmarkDataset

    def init_metrics(self) -> dict:
        return {"total_mnae": 0.0, "count": 0}

    def extract_inputs(self, batch_data) -> torch.Tensor:
        image, gt_kpts, idx = batch_data
        return image

    def process_batch_result(self, batch_data, output, metrics_state: dict) -> dict:
        image, gt_kpts, idx = batch_data

        pred_kpts = output[0]  # [21, 3] or [1, 21, 3]

        if isinstance(pred_kpts, torch.Tensor):
            pred_kpts = pred_kpts.numpy()
        if isinstance(gt_kpts, torch.Tensor):
            gt_kpts = gt_kpts.numpy()

        if pred_kpts.ndim == 3:
            pred_kpts = pred_kpts[0]
        if gt_kpts.ndim == 3:
            gt_kpts = gt_kpts[0]

        pred_xy = pred_kpts[:, :2]
        gt_xy = gt_kpts[:, :2]
        visibility = gt_kpts[:, 2]

        # Palm size: L2 distance from WRIST(0) to MIDDLE_FINGER_MCP(9)
        palm_size = np.sqrt(np.sum((gt_xy[WRIST] - gt_xy[MIDDLE_FINGER_MCP]) ** 2))

        visible_mask = visibility > 0
        if visible_mask.sum() > 0 and palm_size > 1e-6:
            distances = np.sqrt(np.sum((pred_xy[visible_mask] - gt_xy[visible_mask]) ** 2, axis=1))
            mnae = np.mean(distances) / palm_size
        else:
            mnae = 0.0

        metrics_state["total_mnae"] += mnae
        metrics_state["count"] += 1

        return metrics_state

    def compute_final_metrics(self, metrics_state: dict) -> dict:
        count = metrics_state["count"]
        avg_mnae = metrics_state["total_mnae"] / max(count, 1)

        avg_fps = count / self.total_inference_time if self.total_inference_time > 0 else 0.0

        print(f"MNAE: {avg_mnae:.6f}")
        print(f"Average Inference FPS: {avg_fps:.2f}")
        logger.success(f"@JSON <MNAE:{avg_mnae:.6f}; Average FPS:{avg_fps:.2f}>")

        return {
            "performance": [avg_mnae],
            "fps": avg_fps,
        }

    def format_progress_desc(self, metrics_state: dict, current_fps: float) -> str:
        count = metrics_state.get("count", 0)
        avg_mnae = metrics_state["total_mnae"] / max(count, 1) if count > 0 else 0.0
        return f"HandLandmark | MNAE:{avg_mnae:.6f} Count:{count} FPS:{current_fps:.1f}"
