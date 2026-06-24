"""3D object detection evaluator (``3d_object_detection``).

Scores SFA3D-style BEV detections against KITTI ground truth using
**Bird's-Eye-View rotated-box AP**. Predictions come from the
``sfa3d_decode`` postprocessor as a dict
``{boxes, scores, class_ids, extra}`` where ``extra`` carries
``[z, h, w, l, yaw]``; both predictions and GT are reduced to BEV rotated
boxes ``(x, y, l, w, yaw)`` in the LiDAR ground plane and matched with exact
polygon IoU (``cv2.rotatedRectangleIntersection``).

Reported metrics: ``mAP_BEV@0.5`` and ``mAP_BEV@0.7`` (mean over classes).

ponytail: BEV AP, not the full official KITTI 3D AP (no Easy/Mod/Hard split,
no 3D-IoU). It is a single self-contained number that tracks quantization
impact. Swap in the official kitti_eval if leaderboard-comparable numbers are
needed.
"""
from __future__ import annotations

from typing import Any, List, Tuple

import cv2
import numpy as np

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.dataset.kitti import BEV_HEIGHT, BEV_WIDTH, BOUND_SIZE_X, BOUND_SIZE_Y, BOUNDARY
from dx_modelzoo.evaluator import EVALUATOR_REGISTRY, EvaluatorBase
from dx_modelzoo.session import SessionBase


def _rotated_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between two sets of BEV rotated boxes.

    Each box is (cx, cy, l, w, yaw[rad]); returns [len(a), len(b)].
    """
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)

    def _rect(box):
        cx, cy, l, w, yaw = box
        return ((float(cx), float(cy)), (float(l), float(w)), float(np.degrees(yaw)))

    area_a = a[:, 2] * a[:, 3]
    area_b = b[:, 2] * b[:, 3]
    iou = np.zeros((len(a), len(b)), dtype=np.float64)
    rects_b = [_rect(box) for box in b]
    for i, box_a in enumerate(a):
        ra = _rect(box_a)
        for j, rb in enumerate(rects_b):
            ret, region = cv2.rotatedRectangleIntersection(ra, rects_b[j])
            if ret == 0 or region is None:
                continue
            inter = cv2.contourArea(region)
            if inter <= 0:
                continue
            union = area_a[i] + area_b[j] - inter
            if union > 0:
                iou[i, j] = inter / union
    return iou


@EVALUATOR_REGISTRY.register("3d_object_detection")
class ObjectDetection3DEvaluator(EvaluatorBase):
    """BEV rotated-box AP evaluator for SFA3D on KITTI."""

    def __init__(self, session: SessionBase, dataset: DatasetBase, **kwargs) -> None:
        super().__init__(session, dataset, workers=12, **kwargs)
        self.iouv = np.array([0.5, 0.7])
        self.niou = len(self.iouv)
        self.stats: List[dict] = []

    def init_metrics(self) -> dict:
        self.stats = []
        return {"processed": 0}

    def extract_inputs(self, batch_data: Tuple) -> np.ndarray:
        image, _shape, _sample_id = batch_data
        return image

    @staticmethod
    def _preds_to_bev(output: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decode dict -> (bev_boxes [N,5] (x,y,l,w,yaw), conf [N], cls [N])."""
        if output is None or "boxes" not in output or len(output["boxes"]) == 0:
            return np.zeros((0, 5)), np.zeros(0), np.zeros(0)
        boxes = np.asarray(output["boxes"], dtype=np.float64)
        scores = np.asarray(output["scores"], dtype=np.float64)
        cls = np.asarray(output["class_ids"], dtype=np.float64)
        extra = np.asarray(output["extra"], dtype=np.float64)  # [z, h, w, l, yaw]

        # Pixel BEV center recovered from the pseudo xyxy box.
        cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
        cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
        w = extra[:, 2]
        length = extra[:, 3]
        yaw_model = extra[:, 4]

        # Pixel BEV -> real LiDAR ground plane (SFA3D convert_det_to_real_values).
        real_x = cy / BEV_HEIGHT * BOUND_SIZE_X + BOUNDARY["minX"]
        real_y = cx / BEV_WIDTH * BOUND_SIZE_Y + BOUNDARY["minY"]
        yaw = -yaw_model

        bev = np.stack([real_x, real_y, length, w, yaw], axis=1)
        return bev, scores, cls

    def process_batch_result(self, batch_data: Tuple, output: Any, metrics_state: dict) -> dict:
        _image, _shape, sample_id = batch_data
        pred_bev, conf, pred_cls = self._preds_to_bev(output)

        gt_boxes, gt_cls = self.dataset.get_gt(sample_id)
        if len(gt_boxes):
            # (x, y, z, h, w, l, yaw) -> BEV (x, y, l, w, yaw)
            gt_bev = gt_boxes[:, [0, 1, 5, 4, 6]]
        else:
            gt_bev = np.zeros((0, 5))

        self.stats.append(self._match(pred_bev, conf, pred_cls, gt_bev, gt_cls))
        metrics_state["processed"] += 1
        return metrics_state

    def _match(self, pred_bev, conf, pred_cls, gt_bev, gt_cls) -> dict:
        if len(gt_cls) == 0 or len(pred_cls) == 0:
            return {
                "tp": np.zeros((len(pred_cls), self.niou), dtype=bool),
                "conf": conf,
                "pred_cls": pred_cls,
                "target_cls": gt_cls,
            }
        iou = _rotated_iou(gt_bev, pred_bev)  # [n_gt, n_pred]
        correct = np.zeros((len(pred_cls), self.niou), dtype=bool)
        correct_class = gt_cls[:, None] == pred_cls[None, :]
        iou = iou * correct_class
        for i, thr in enumerate(self.iouv):
            matches = np.argwhere(iou >= thr)
            if matches.shape[0]:
                if matches.shape[0] > 1:
                    matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True
        return {"tp": correct, "conf": conf, "pred_cls": pred_cls, "target_cls": gt_cls}

    def compute_final_metrics(self, metrics_state: dict) -> dict:
        avg_fps = metrics_state["processed"] / self.total_inference_time if self.total_inference_time > 0 else 0.0
        if not self.stats:
            return self._finalize(["mAP_BEV@0.5", "mAP_BEV@0.7"], [0.0, 0.0], avg_fps)

        merged = {k: np.concatenate([s[k] for s in self.stats], 0) for k in self.stats[0]}
        map50 = map70 = 0.0
        if len(merged["tp"]):
            ap = self._ap_per_class(merged["tp"], merged["conf"], merged["pred_cls"], merged["target_cls"])
            if ap.size:
                map50 = float(ap[:, 0].mean())
                map70 = float(ap[:, 1].mean())
        return self._finalize(["mAP_BEV@0.5", "mAP_BEV@0.7"], [map50 * 100, map70 * 100], avg_fps)

    def format_progress_desc(self, metrics_state: dict, current_fps: float) -> str:
        return f"3D-Det | Frames:{metrics_state.get('processed', 0)} Current_FPS:{current_fps:.1f}"

    def _ap_per_class(self, tp, conf, pred_cls, target_cls) -> np.ndarray:
        order = np.argsort(-conf)
        tp, pred_cls = tp[order], pred_cls[order]
        unique_classes, nt = np.unique(target_cls, return_counts=True)
        ap = np.zeros((len(unique_classes), tp.shape[1]))
        for ci, c in enumerate(unique_classes):
            i = pred_cls == c
            n_l = nt[ci]
            if i.sum() == 0 or n_l == 0:
                continue
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)
            recall = tpc / (n_l + 1e-16)
            precision = tpc / (tpc + fpc)
            for j in range(tp.shape[1]):
                ap[ci, j] = self._compute_ap(recall[:, j], precision[:, j])
        return ap

    @staticmethod
    def _compute_ap(recall, precision) -> float:
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
        x = np.linspace(0, 1, 101)
        _trapz = getattr(np, "trapezoid", None) or np.trapz
        return float(_trapz(np.interp(x, mrec, mpre), x))
