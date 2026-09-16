from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from dx_modelzoo.evaluator import EVALUATOR_REGISTRY, EvaluatorBase
from dx_modelzoo.session import SessionBase


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    areas = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    return intersection / np.maximum(area + areas - intersection, 1e-9)


def top1_hit(predictions: np.ndarray, ground_truth: np.ndarray, iou_threshold: float = 0.5) -> bool:
    """Return whether the highest-confidence prediction overlaps any GT box."""
    predictions = np.asarray(predictions)
    ground_truth = np.asarray(ground_truth)
    if predictions.ndim == 3:
        predictions = predictions[0]
    if predictions.ndim != 2 or predictions.shape[1] < 5:
        raise ValueError(f"Predictions must have shape [N, 5+], got {predictions.shape}")
    if ground_truth.ndim != 2 or ground_truth.shape[1] != 4:
        raise ValueError(f"Ground truth boxes must have shape [N, 4], got {ground_truth.shape}")
    if not len(predictions) or not len(ground_truth):
        return False
    best = predictions[np.argmax(predictions[:, 4]), :4]
    return bool(np.max(_iou(best, ground_truth)) >= iou_threshold)


def _ground_truth_boxes(dataset: Any, image_id: int) -> np.ndarray:
    coco = getattr(dataset, "coco_annotation", None)
    if coco is None:
        raise ValueError("mainbody_detection requires a COCO-compatible dataset with annotations.")
    annotations: Iterable[dict] = coco.loadAnns(coco.getAnnIds(imgIds=[image_id]))
    boxes = [
        [ann["bbox"][0], ann["bbox"][1], ann["bbox"][0] + ann["bbox"][2], ann["bbox"][1] + ann["bbox"][3]]
        for ann in annotations
        if not ann.get("iscrowd", 0) and ann.get("bbox")
    ]
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


@EVALUATOR_REGISTRY.register("mainbody_detection")
class MainbodyDetectionEvaluator(EvaluatorBase):
    """Top-1 class-agnostic localization evaluator for mainbody detectors."""

    def __init__(self, session: SessionBase, dataset: Any, iou_threshold: float = 0.5, **kwargs: Any) -> None:
        super().__init__(session, dataset, **kwargs)
        self.iou_threshold = float(iou_threshold)

    def init_metrics(self) -> dict:
        return {"hits": 0, "images": 0}

    def extract_inputs(self, batch_data: tuple) -> np.ndarray:
        image = batch_data[0]
        return np.expand_dims(image, 0) if image.ndim == 3 else image

    def process_batch_result(self, batch_data: tuple, output: Any, metrics_state: dict) -> dict:
        image_id = int(batch_data[2])
        predictions = np.asarray(output)
        ground_truth = _ground_truth_boxes(self.dataset, image_id)
        metrics_state["hits"] += int(top1_hit(predictions, ground_truth, self.iou_threshold))
        metrics_state["images"] += 1
        return metrics_state

    def compute_final_metrics(self, metrics_state: dict) -> dict:
        images = metrics_state["images"]
        value = 100.0 * metrics_state["hits"] / images if images else 0.0
        fps = images / self.total_inference_time if self.total_inference_time > 0 else 0.0
        return self._finalize(metric_names=["Recall@1"], metric_values=[value], fps=fps)

    def format_progress_desc(self, metrics_state: dict, current_fps: float) -> str:
        return f"Mainbody Recall@1 | Hits:{metrics_state['hits']}/{metrics_state['images']} FPS:{current_fps:.1f}"
