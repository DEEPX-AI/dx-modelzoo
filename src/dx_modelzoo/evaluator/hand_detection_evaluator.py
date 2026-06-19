from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.evaluator import EVALUATOR_REGISTRY, EvaluatorBase
from dx_modelzoo.session import SessionBase


def _iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU of a single ``xyxy`` box against an ``[N, 4]`` array of boxes."""
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return inter / (area + areas - inter + 1e-9)


@EVALUATOR_REGISTRY.register("hand_detection")
class HandDetectionEvaluator(EvaluatorBase):
    """Single-class hand detection evaluator computing AP@0.5 (VOC all-point)."""

    iou_thresh: float = 0.5

    def __init__(self, session: SessionBase, dataset: DatasetBase, **kwargs) -> None:
        super().__init__(session, dataset, workers=12, **kwargs)

    def init_metrics(self) -> dict:
        return {"scores": [], "tp": [], "n_gt": 0}

    def extract_inputs(self, batch_data: Tuple) -> np.ndarray:
        image, _boxes, _origin_hw, _path = batch_data
        return image

    def process_batch_result(self, batch_data: Tuple, output: Any, metrics_state: dict) -> dict:
        _image, gt_boxes, _origin_hw, _path = batch_data
        gt = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 4)

        dets = output if isinstance(output, (list, tuple)) else [output]
        dets = np.asarray(dets, dtype=np.float32).reshape(-1, 5) if len(dets) > 0 else np.zeros((0, 5), np.float32)

        metrics_state["n_gt"] += len(gt)

        # Greedy matching of predictions (highest score first) to GT boxes.
        order = np.argsort(-dets[:, 4]) if len(dets) else np.array([], dtype=int)
        matched = np.zeros(len(gt), dtype=bool)
        for i in order:
            d = dets[i]
            tp = 0
            if len(gt) > 0:
                ious = _iou_one_to_many(d[:4], gt)
                best = int(np.argmax(ious))
                if ious[best] >= self.iou_thresh and not matched[best]:
                    matched[best] = True
                    tp = 1
            metrics_state["scores"].append(float(d[4]))
            metrics_state["tp"].append(tp)
        return metrics_state

    @staticmethod
    def _voc_ap(rec: np.ndarray, prec: np.ndarray) -> float:
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

    def _compute_ap(self, metrics_state: dict) -> float:
        n_gt = metrics_state["n_gt"]
        scores = np.asarray(metrics_state["scores"], dtype=np.float32)
        tp_flags = np.asarray(metrics_state["tp"], dtype=np.float32)
        if n_gt == 0 or len(scores) == 0:
            return 0.0
        order = np.argsort(-scores)
        tp_flags = tp_flags[order]
        fp_flags = 1.0 - tp_flags
        tp_cum = np.cumsum(tp_flags)
        fp_cum = np.cumsum(fp_flags)
        recall = tp_cum / max(n_gt, 1)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        return self._voc_ap(recall, precision)

    def compute_final_metrics(self, metrics_state: dict) -> dict:
        total_len = len(self.dataset)
        avg_fps = total_len / self.total_inference_time if self.total_inference_time > 0 else 0.0
        ap = self._compute_ap(metrics_state)
        return self._finalize(
            metric_names=["AP@0.5"],
            metric_values=[ap * 100.0],
            fps=avg_fps,
        )

    def format_progress_desc(self, metrics_state: dict, current_fps: float) -> str:
        ap = self._compute_ap(metrics_state)
        return f"HandDetection | AP@0.5:{ap*100:.2f} GT:{metrics_state['n_gt']} FPS:{current_fps:.1f}"
