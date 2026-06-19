from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
from loguru import logger

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.evaluator import EVALUATOR_REGISTRY, EvaluatorBase
from dx_modelzoo.session import SessionBase


@EVALUATOR_REGISTRY.register("zero_shot_instance_segmentation")
class ZeroShotInstanceSegmentationEvaluator(EvaluatorBase):
    """COCO Evaluator for zero-shot instance segmentation (AR metrics)."""

    def __init__(self, session: SessionBase, dataset: DatasetBase, **kwargs) -> None:
        super().__init__(session, dataset, workers=12, **kwargs)
        self._detections: List[dict] = []

    def init_metrics(self) -> dict:
        self._detections = []
        return {"detections_count": 0, "image_count": 0}

    def extract_inputs(self, batch_data: Tuple) -> np.ndarray:
        image, origin_shape_tensors, img_id_tensor = batch_data
        return image

    def _build_postprocessing_context(self, batch_data) -> dict:
        image, origin_shape_tensors, img_id_tensor = batch_data
        origin_hw = (int(origin_shape_tensors[0]), int(origin_shape_tensors[1]))
        if image.ndim >= 3 and image.shape[-1] in (1, 3):
            input_hw = (image.shape[-3], image.shape[-2])
        else:
            input_hw = (image.shape[-2], image.shape[-1])
        return {"origin_hw": origin_hw, "input_hw": input_hw}

    def process_batch_result(self, batch_data: Tuple, output: Any, metrics_state: dict) -> dict:
        image, origin_shape_tensors, img_id_tensor = batch_data
        img_id = int(img_id_tensor)

        outputs, final_masks = output
        if outputs is not None and len(outputs) > 0:
            outputs = np.asarray(outputs)
            final_masks = np.asarray(final_masks)
            coco_dets = self._make_coco_format_seg(final_masks, outputs, img_id)
            if coco_dets:
                self._detections.extend(coco_dets)
                metrics_state["detections_count"] += len(coco_dets)
        metrics_state["image_count"] += 1
        return metrics_state

    def compute_final_metrics(self, metrics_state: dict) -> dict:
        ar10, ar100, ar1000 = self._run_coco_eval()
        avg_fps = metrics_state["image_count"] / self.total_inference_time if self.total_inference_time > 0 else 0.0
        return self._finalize(
            metric_names=["AR@10", "AR@100", "AR@1000"],
            metric_values=[ar10 * 100, ar100 * 100, ar1000 * 100],
            fps=avg_fps,
        )

    def format_progress_desc(self, metrics_state: dict, current_fps: float) -> str:
        det_count = metrics_state.get("detections_count", 0)
        return f"COCO | Dets:{det_count} Current_FPS:{current_fps:.1f}"

    def _make_coco_format_seg(self, masks: np.ndarray, outputs: np.ndarray, img_id: int) -> List[dict]:
        try:
            from faster_coco_eval.core.mask import encode as rle_encode
        except ImportError:
            return []
        seg_detections = []
        for i in range(len(outputs)):
            score = round(float(outputs[i][4]), 5)
            rle = rle_encode(np.asfortranarray(masks[i]))
            rle["counts"] = rle["counts"].decode("utf-8")
            seg_detections.append(
                {
                    "image_id": img_id,
                    "category_id": 1,
                    "segmentation": rle,
                    "score": score,
                }
            )
        return seg_detections

    def _run_coco_eval(self) -> Tuple[float, float, float]:
        try:
            from faster_coco_eval import COCOeval_faster

            if not self._detections:
                return 0.0, 0.0, 0.0
            predicted = self.dataset.coco_annotation.loadRes(self._detections)
            coco_eval = COCOeval_faster(self.dataset.coco_annotation, predicted, "segm")
            coco_eval.params.useCats = 0
            coco_eval.params.maxDets = [10, 100, 1000]
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            return coco_eval.stats[6], coco_eval.stats[7], coco_eval.stats[8]
        except Exception as e:
            logger.error("Error during COCO evaluation: {}", e)
            return 0.0, 0.0, 0.0
