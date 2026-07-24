from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
from loguru import logger

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.evaluator import EVALUATOR_REGISTRY, EvaluatorBase
from dx_modelzoo.session import SessionBase


@EVALUATOR_REGISTRY.register("object_detection")
class ObjectDetectionEvaluator(EvaluatorBase):
    """Unified Object Detection Evaluator — outputs mAP and mAP50.

    Works with any detection dataset that provides ``coco_annotation`` and
    ``remap_class_id()`` (e.g. COCO, PascalVOC2007).
    """

    def __init__(self, session: SessionBase, dataset: DatasetBase, **kwargs) -> None:
        super().__init__(session, dataset, workers=12, **kwargs)
        self._detections: List[dict] = []

    def init_metrics(self) -> dict:
        self._detections = []
        return {"detections_count": 0, "image_count": 0}

    def extract_inputs(self, batch_data: Tuple) -> np.ndarray:
        image = batch_data[0]
        if image.ndim == 3:
            image = np.expand_dims(image, 0)
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
        img_id = int(batch_data[2])

        metrics_state["image_count"] += 1

        if isinstance(output, np.ndarray) and output.shape[0] > 0:
            detections = self._format_detections(output, img_id)
            if detections:
                self._detections.extend(detections)
                metrics_state["detections_count"] += len(detections)

        return metrics_state

    def compute_final_metrics(self, metrics_state: dict) -> dict:
        mAP, mAP50 = 0.0, 0.0
        try:
            from faster_coco_eval import COCOeval_faster

            coco_gt = self.dataset.coco_annotation
            if self._detections and coco_gt is not None:
                predicted = coco_gt.loadRes(self._detections)
                coco_eval = COCOeval_faster(coco_gt, predicted, "bbox")
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                mAP, mAP50 = coco_eval.stats[:2]
        except Exception as e:
            logger.error("Object detection evaluation failed: {}", e)

        avg_fps = metrics_state["image_count"] / self.total_inference_time if self.total_inference_time > 0 else 0

        logger.info("mAP: {} mAP50: {}", round(mAP * 100, 3), round(mAP50 * 100, 3))
        return self._finalize(
            metric_names=["mAP", "mAP50"],
            metric_values=[mAP * 100, mAP50 * 100],
            fps=avg_fps,
        )

    def format_progress_desc(self, metrics_state: dict, current_fps: float) -> str:
        img_count = metrics_state.get("image_count", 0)
        det_count = metrics_state.get("detections_count", 0)
        return f"Detection | Imgs:{img_count} Dets:{det_count} FPS:{current_fps:.1f}"

    def _format_detections(self, output: np.ndarray, img_id: int) -> List[dict]:
        if output.shape[0] == 1 and len(output.shape) == 3:
            output = output[0]

        detections = []
        for row in output:
            x1, y1, x2, y2, conf, cls_id = row[:6]
            w, h = x2 - x1, y2 - y1
            category_id = (
                int(cls_id) if getattr(self, "use_class_90", False) else self.dataset.remap_class_id(int(cls_id))
            )
            det = {
                "image_id": int(img_id),
                "category_id": category_id,
                "bbox": [round(float(x1), 3), round(float(y1), 3), round(float(w), 3), round(float(h), 3)],
                "score": round(float(conf), 5),
            }
            detections.append(det)
        return detections
