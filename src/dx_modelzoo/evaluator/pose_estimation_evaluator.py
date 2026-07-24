from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
from loguru import logger

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.evaluator import EVALUATOR_REGISTRY, EvaluatorBase
from dx_modelzoo.session import SessionBase


@EVALUATOR_REGISTRY.register("pose_estimation")
class PoseEstimationEvaluator(EvaluatorBase):
    """COCO Pose Evaluator for keypoint detection (mAP)."""

    def __init__(self, session: SessionBase, dataset: DatasetBase, **kwargs) -> None:
        super().__init__(session, dataset, workers=12, **kwargs)
        self._detections: List[dict] = []

    def init_metrics(self) -> dict:
        self._detections = []
        return {"detections_count": 0, "image_count": 0}

    def extract_inputs(self, batch_data: Tuple) -> np.ndarray:
        image, origin_shape, img_id = batch_data
        return image

    def _build_postprocessing_context(self, batch_data) -> dict:
        image, origin_shape, _id = batch_data
        if isinstance(origin_shape, (list, tuple)):
            origin_shape = [int(v[0]) if hasattr(v, "__getitem__") else int(v) for v in origin_shape]
        origin_hw = (int(origin_shape[0]), int(origin_shape[1]))
        if image.ndim >= 3 and image.shape[-1] in (1, 3):
            input_hw = (image.shape[-3], image.shape[-2])
        else:
            input_hw = (image.shape[-2], image.shape[-1])
        return {"origin_hw": origin_hw, "input_hw": input_hw}

    def process_batch_result(self, batch_data: Tuple, output: Any, metrics_state: dict) -> dict:
        image, origin_shape, img_id = batch_data

        boxes, scores, _cls, keypoints = output
        if not isinstance(boxes, np.ndarray):
            boxes = np.asarray(boxes)
        if not isinstance(keypoints, np.ndarray):
            keypoints = np.asarray(keypoints)
        scores = np.asarray(scores)

        if len(boxes) > 0:
            coco_dets = self._make_coco_format_pose(boxes, scores, keypoints, img_id)
            if coco_dets:
                self._detections.extend(coco_dets)
                metrics_state["detections_count"] += len(coco_dets)
        metrics_state["image_count"] += 1
        return metrics_state

    def compute_final_metrics(self, metrics_state: dict) -> dict:
        mAP, mAP50 = self._run_coco_eval()
        avg_fps = metrics_state["image_count"] / self.total_inference_time if self.total_inference_time > 0 else 0.0
        return self._finalize(
            metric_names=["mAP", "mAP50"],
            metric_values=[mAP * 100, mAP50 * 100],
            fps=avg_fps,
        )

    def format_progress_desc(self, metrics_state: dict, current_fps: float) -> str:
        det_count = metrics_state.get("detections_count", 0)
        return f"COCO | Dets:{det_count} Current_FPS:{current_fps:.1f}"

    def _make_coco_format_pose(
        self, boxes: np.ndarray, scores: np.ndarray, keypoints: np.ndarray, img_id: Any
    ) -> List[dict]:
        detections = []
        image_id = int(img_id.item() if hasattr(img_id, "item") else img_id)
        for i in range(len(scores)):
            x1, y1, x2, y2 = boxes[i][:4]
            detection = {
                "image_id": image_id,
                "category_id": 1,
                "bbox": [round(float(x1), 3), round(float(y1), 3), round(float(x2 - x1), 3), round(float(y2 - y1), 3)],
                "score": round(float(scores[i]), 5),
                "keypoints": [round(c, 3) for c in keypoints[i].flatten().tolist()],
            }
            detections.append(detection)
        return detections

    def _run_coco_eval(self) -> Tuple[float, float]:
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
