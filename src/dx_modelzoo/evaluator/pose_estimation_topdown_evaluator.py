"""Top-down pose estimation evaluator -- COCO AP (OKS).

Requires the ``COCOPoseTopDown`` dataset which provides GT person crops
with crop mapping info for back-projecting predictions to image coords.
"""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
from loguru import logger

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.evaluator import EVALUATOR_REGISTRY, EvaluatorBase
from dx_modelzoo.session import SessionBase

# SimCC split ratio (standard default for RTMPose)
_SIMCC_SPLIT_RATIO = 2.0

# BlazePose 33/39 -> COCO 17 landmark index mapping
_BLAZEPOSE_TO_COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


def _decode_simcc(
    simcc_x: np.ndarray, simcc_y: np.ndarray, model_w: float, model_h: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Decode SimCC heatmaps to keypoint coordinates and scores.

    Args:
        simcc_x: ``[1, 17, W_bins]``
        simcc_y: ``[1, 17, H_bins]``

    Returns:
        keypoints ``[17, 2]`` (x, y) in model-input pixel coords,
        scores ``[17]``.
    """
    x_idx = simcc_x[0].argmax(axis=-1).astype(np.float32)  # [17]
    y_idx = simcc_y[0].argmax(axis=-1).astype(np.float32)  # [17]
    x_score = simcc_x[0].max(axis=-1)
    y_score = simcc_y[0].max(axis=-1)
    scores = (x_score + y_score) / 2.0
    kpts = np.stack([x_idx / _SIMCC_SPLIT_RATIO, y_idx / _SIMCC_SPLIT_RATIO], axis=-1)
    return kpts, scores


def _decode_direct(raw_landmarks: np.ndarray, landmark_mapping: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Decode direct landmark coordinates (e.g. MediaPipePose).

    Args:
        raw_landmarks: ``[1, N*5]`` or ``[N, 5]`` with (x, y, z, vis, pres).
        landmark_mapping: Source landmark indices to extract for COCO 17.

    Returns:
        keypoints ``[17, 2]`` (x, y) in model-input pixel coords,
        scores ``[17]``.
    """
    lm = np.asarray(raw_landmarks).flatten()
    num_lm = len(lm) // 5
    lm = lm.reshape(num_lm, 5)

    kpts_17 = np.zeros((17, 2), dtype=np.float32)
    scores_17 = np.zeros(17, dtype=np.float32)
    for coco_idx, src_idx in enumerate(landmark_mapping):
        if src_idx < num_lm:
            kpts_17[coco_idx, 0] = lm[src_idx, 0]
            kpts_17[coco_idx, 1] = lm[src_idx, 1]
            vis = lm[src_idx, 3]
            scores_17[coco_idx] = 1.0 / (1.0 + np.exp(-vis))

    return kpts_17, scores_17


@EVALUATOR_REGISTRY.register("pose_estimation_topdown")
class PoseEstimationTopDownEvaluator(EvaluatorBase):
    """COCO AP (OKS-based) evaluator for top-down pose models.

    Postprocessing delivers ``(kpts_pred [17, 2], scores [17])`` in
    model-input pixel coordinates.
    """

    # Configurable via evaluator.options
    model_input_w: float = 192.0
    model_input_h: float = 256.0

    def __init__(self, session: SessionBase, dataset: DatasetBase, **kwargs) -> None:
        super().__init__(session, dataset, **kwargs)
        self._detections: List[dict] = []

    def init_metrics(self) -> dict:
        self._detections = []
        return {"detections_count": 0, "sample_count": 0}

    def extract_inputs(self, batch_data: Tuple) -> np.ndarray:
        image = batch_data[0]
        if isinstance(image, np.ndarray) and image.ndim == 3:
            image = np.expand_dims(image, 0)
        return image

    def process_batch_result(self, batch_data: Tuple, output: Any, metrics_state: dict) -> dict:
        metrics_state["sample_count"] += 1
        gt_bbox = batch_data[1]
        crop_params = batch_data[2]
        visibility = np.asarray(batch_data[4])
        img_id = batch_data[5]

        kpts_pred, scores = output

        crop_x1 = float(crop_params[0])
        crop_y1 = float(crop_params[1])
        crop_w = float(crop_params[2])
        crop_h = float(crop_params[3])

        pred_x_orig = kpts_pred[:, 0] * (crop_w / self.model_input_w) + crop_x1
        pred_y_orig = kpts_pred[:, 1] * (crop_h / self.model_input_h) + crop_y1

        pred_vis = np.where(visibility > 0, 1, 0).astype(np.float32)

        coco_kpts = np.zeros(17 * 3, dtype=np.float64)
        coco_kpts[0::3] = pred_x_orig
        coco_kpts[1::3] = pred_y_orig
        coco_kpts[2::3] = pred_vis

        vis_mask = visibility > 0
        mean_score = float(scores[vis_mask].mean()) if vis_mask.any() else float(scores.mean())

        bx, by, bw, bh = (float(v) for v in gt_bbox)
        detection = {
            "image_id": int(img_id),
            "category_id": 1,
            "keypoints": [round(c, 3) for c in coco_kpts.tolist()],
            "score": round(mean_score, 5),
            "bbox": [round(bx, 3), round(by, 3), round(bw, 3), round(bh, 3)],
        }
        self._detections.append(detection)
        metrics_state["detections_count"] += 1
        return metrics_state

    def compute_final_metrics(self, metrics_state: dict) -> dict:
        ap, ap50 = self._run_coco_eval()
        n = metrics_state["sample_count"]
        fps = n / self.total_inference_time if self.total_inference_time > 0 else 0
        return self._finalize(metric_names=["mAP", "mAP50"], metric_values=[ap * 100, ap50 * 100], fps=fps)

    def format_progress_desc(self, metrics_state: dict, current_fps: float) -> str:
        det_count = metrics_state.get("detections_count", 0)
        return f"COCO-Pose | Dets:{det_count} FPS:{current_fps:.1f}"

    def _run_coco_eval(self) -> Tuple[float, float]:
        """Run COCOeval_faster with iouType='keypoints' and return (AP, AP50)."""
        try:
            from faster_coco_eval import COCOeval_faster

            if not self._detections:
                return 0.0, 0.0
            coco_ann = self.dataset.coco_annotation
            predicted = coco_ann.loadRes(self._detections)
            coco_eval = COCOeval_faster(coco_ann, predicted, "keypoints")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            return coco_eval.stats[0], coco_eval.stats[1]
        except Exception as e:
            logger.error("COCO pose evaluation failed: {}", e)
            return 0.0, 0.0
