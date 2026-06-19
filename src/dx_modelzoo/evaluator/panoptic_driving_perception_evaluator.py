from __future__ import annotations

from typing import Any, Tuple

import cv2
import numpy as np

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.evaluator import EVALUATOR_REGISTRY, EvaluatorBase
from dx_modelzoo.session import SessionBase


def _box_iou_matrix(preds: np.ndarray, gts: np.ndarray) -> np.ndarray:
    """IoU between ``[N, 4]`` preds and ``[M, 4]`` gts (xyxy). Returns ``[N, M]``."""
    if preds.shape[0] == 0 or gts.shape[0] == 0:
        return np.zeros((preds.shape[0], gts.shape[0]), dtype=np.float32)
    p = preds[:, None, :]
    g = gts[None, :, :]
    inter_w = np.clip(np.minimum(p[..., 2], g[..., 2]) - np.maximum(p[..., 0], g[..., 0]), 0, None)
    inter_h = np.clip(np.minimum(p[..., 3], g[..., 3]) - np.maximum(p[..., 1], g[..., 1]), 0, None)
    inter = inter_w * inter_h
    area_p = (preds[:, 2] - preds[:, 0]) * (preds[:, 3] - preds[:, 1])
    area_g = (gts[:, 2] - gts[:, 0]) * (gts[:, 3] - gts[:, 1])
    union = area_p[:, None] + area_g[None, :] - inter
    return inter / np.clip(union, 1e-9, None)


def _compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """COCO-style 101-point interpolated average precision."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(np.trapz(np.interp(x, mrec, mpre), x))


@EVALUATOR_REGISTRY.register("panoptic_driving_perception")
class PanopticDrivingPerceptionEvaluator(EvaluatorBase):
    """YOLOPv2 panoptic driving perception evaluator.

    Computes, simultaneously over BDD100K:
      * vehicle detection ``mAP@0.5`` and ``Recall`` (single class),
      * drivable area ``mIoU`` (mean over background/drivable),
      * lane line ``Accuracy`` (foreground recall) and ``IoU``.

    Segmentation metrics are computed per image and averaged (YOLOP-style) at
    the model content resolution; the decoded prediction masks define that
    resolution and the ground truth is resized to match.
    """

    iou_match_thres: float = 0.5

    def __init__(self, session: SessionBase, dataset: DatasetBase, **kwargs) -> None:
        super().__init__(session, dataset, workers=12, **kwargs)

    def init_metrics(self) -> dict:
        return {
            "det_scores": [],
            "det_tp": [],
            "det_n_gt": 0,
            "da_miou_sum": 0.0,
            "da_count": 0,
            "ll_acc_sum": 0.0,
            "ll_iou_sum": 0.0,
            "ll_count": 0,
        }

    def extract_inputs(self, batch_data: Tuple) -> np.ndarray:
        image = batch_data[0]
        if image.ndim == 3:
            image = np.expand_dims(image, 0)
        return image

    def _build_postprocessing_context(self, batch_data) -> dict:
        image, origin_shape, _label = batch_data
        origin_hw = (int(origin_shape[0]), int(origin_shape[1]))
        if image.ndim >= 3 and image.shape[-1] in (1, 3):
            input_hw = (image.shape[-3], image.shape[-2])
        else:
            input_hw = (image.shape[-2], image.shape[-1])
        return {"origin_hw": origin_hw, "input_hw": input_hw}

    def process_batch_result(self, batch_data: Tuple, output: Any, metrics_state: dict) -> dict:
        _image, _origin_shape, label = batch_data

        self._update_detection(output.get("boxes"), label["boxes"], metrics_state)
        self._update_drivable(output.get("drivable"), label["drivable"], metrics_state)
        self._update_lane(output.get("lane"), label["lane"], metrics_state)
        return metrics_state

    def _update_detection(self, preds: np.ndarray, gts: np.ndarray, state: dict) -> None:
        gts = np.asarray(gts, dtype=np.float32).reshape(-1, 4)
        state["det_n_gt"] += int(gts.shape[0])
        if preds is None or len(preds) == 0:
            return
        preds = np.asarray(preds, dtype=np.float32)
        order = preds[:, 4].argsort()[::-1]
        preds = preds[order]

        tp = np.zeros(preds.shape[0], dtype=np.float32)
        if gts.shape[0] > 0:
            iou = _box_iou_matrix(preds[:, :4], gts)
            matched = set()
            for i in range(preds.shape[0]):
                j = int(np.argmax(iou[i]))
                if iou[i, j] >= self.iou_match_thres and j not in matched:
                    tp[i] = 1.0
                    matched.add(j)
        state["det_scores"].append(preds[:, 4].astype(np.float32))
        state["det_tp"].append(tp)

    @staticmethod
    def _seg_confusion(pred: np.ndarray, gt: np.ndarray) -> Tuple[int, int, int, int]:
        """Foreground (class 1) TP, FP, FN, TN."""
        p1 = pred == 1
        g1 = gt == 1
        tp = int(np.count_nonzero(p1 & g1))
        fp = int(np.count_nonzero(p1 & ~g1))
        fn = int(np.count_nonzero(~p1 & g1))
        tn = int(np.count_nonzero(~p1 & ~g1))
        return tp, fp, fn, tn

    def _update_drivable(self, pred: np.ndarray, gt: np.ndarray, state: dict) -> None:
        if pred is None or pred.size <= 1:
            return
        gt_resized = self._resize_gt(gt, pred.shape)
        tp, fp, fn, tn = self._seg_confusion(pred, gt_resized)
        iou_fg = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0
        iou_bg = tn / (tn + fp + fn) if (tn + fp + fn) > 0 else 1.0
        state["da_miou_sum"] += 0.5 * (iou_fg + iou_bg)
        state["da_count"] += 1

    def _update_lane(self, pred: np.ndarray, gt: np.ndarray, state: dict) -> None:
        if pred is None or pred.size <= 1:
            return
        gt_resized = self._resize_gt(gt, pred.shape)
        tp, fp, fn, _tn = self._seg_confusion(pred, gt_resized)
        acc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        state["ll_acc_sum"] += acc
        state["ll_iou_sum"] += iou
        state["ll_count"] += 1

    @staticmethod
    def _resize_gt(gt: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        gt = np.asarray(gt)
        th, tw = target_shape[0], target_shape[1]
        if gt.shape[0] == th and gt.shape[1] == tw:
            return gt
        return cv2.resize(gt.astype(np.uint8), (tw, th), interpolation=cv2.INTER_NEAREST).astype(np.int64)

    def compute_final_metrics(self, metrics_state: dict) -> dict:
        map50, recall = self._compute_detection_metrics(metrics_state)

        da_count = metrics_state["da_count"]
        drivable_miou = (metrics_state["da_miou_sum"] / da_count * 100) if da_count else 0.0

        ll_count = metrics_state["ll_count"]
        lane_acc = (metrics_state["ll_acc_sum"] / ll_count * 100) if ll_count else 0.0
        lane_iou = (metrics_state["ll_iou_sum"] / ll_count * 100) if ll_count else 0.0

        avg_fps = len(self.dataset) / self.total_inference_time if self.total_inference_time > 0 else 0.0
        return self._finalize(
            metric_names=[
                "det_mAP50",
                "det_Recall",
                "drivable_mIoU",
                "lane_Acc",
                "lane_IoU",
            ],
            metric_values=[map50, recall, drivable_miou, lane_acc, lane_iou],
            fps=avg_fps,
        )

    def _compute_detection_metrics(self, state: dict) -> Tuple[float, float]:
        n_gt = state["det_n_gt"]
        if n_gt == 0 or not state["det_scores"]:
            return 0.0, 0.0
        scores = np.concatenate(state["det_scores"])
        tp = np.concatenate(state["det_tp"])
        order = scores.argsort()[::-1]
        tp = tp[order]
        fp = 1.0 - tp
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall_curve = tp_cum / (n_gt + 1e-9)
        precision_curve = tp_cum / np.clip(tp_cum + fp_cum, 1e-9, None)
        ap = _compute_ap(recall_curve, precision_curve)
        recall = float(recall_curve[-1]) if recall_curve.size else 0.0
        return ap * 100, recall * 100

    def format_progress_desc(self, metrics_state: dict, current_fps: float) -> str:
        return (
            f"Panoptic | GT_veh:{metrics_state['det_n_gt']} "
            f"DA:{metrics_state['da_count']} LL:{metrics_state['ll_count']} "
            f"FPS:{current_fps:.1f}"
        )
