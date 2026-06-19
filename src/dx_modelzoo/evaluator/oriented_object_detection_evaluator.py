from __future__ import annotations

from typing import Any, List, Tuple

import cv2
import numpy as np

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.evaluator import EVALUATOR_REGISTRY, EvaluatorBase
from dx_modelzoo.session import SessionBase


def _get_covariance_matrix(boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute covariance matrix components from oriented bounding boxes.

    Args:
        boxes: (N, 5) array with [cx, cy, w, h, angle] (xywhr format).

    Returns:
        (a, b, c) covariance components where matrix is [[a, c], [c, b]].
    """
    # Gaussian bounding box representation
    w_sq = boxes[:, 2:3] ** 2 / 12
    h_sq = boxes[:, 3:4] ** 2 / 12
    angle = boxes[:, 4:5]
    cos = np.cos(angle)
    sin = np.sin(angle)
    cos2 = cos**2
    sin2 = sin**2
    return (
        w_sq * cos2 + h_sq * sin2,
        w_sq * sin2 + h_sq * cos2,
        (w_sq - h_sq) * cos * sin,
    )


def batch_probiou(obb1: np.ndarray, obb2: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Calculate probabilistic IoU between two sets of oriented bounding boxes.

    Based on https://arxiv.org/pdf/2106.06072v1.pdf — uses Bhattacharyya distance
    between Gaussian representations of the OBBs.

    Args:
        obb1: (N, 5) ground truth OBBs in xywhr format.
        obb2: (M, 5) predicted OBBs in xywhr format.
        eps: Small value to avoid division by zero.

    Returns:
        (N, M) IoU similarity matrix.
    """
    x1 = obb1[:, 0:1]  # (N, 1)
    y1 = obb1[:, 1:2]  # (N, 1)
    x2 = obb2[:, 0:1].T  # (1, M)
    y2 = obb2[:, 1:2].T  # (1, M)

    a1, b1, c1 = _get_covariance_matrix(obb1)  # each (N, 1)
    a2, b2, c2 = _get_covariance_matrix(obb2)  # each (M, 1)
    a2, b2, c2 = a2.T, b2.T, c2.T  # each (1, M)

    t1 = (
        ((a1 + a2) * (y1 - y2) ** 2 + (b1 + b2) * (x1 - x2) ** 2) / ((a1 + a2) * (b1 + b2) - (c1 + c2) ** 2 + eps)
    ) * 0.25

    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2) ** 2 + eps)) * 0.5

    det1 = np.maximum(a1 * b1 - c1**2, 0)
    det2 = np.maximum(a2 * b2 - c2**2, 0)
    t3 = np.log(((a1 + a2) * (b1 + b2) - (c1 + c2) ** 2) / (4 * np.sqrt(det1 * det2) + eps) + eps) * 0.5

    bd = np.clip(t1 + t2 + t3, eps, 100.0)
    hd = np.sqrt(1.0 - np.exp(-bd) + eps)
    return 1 - hd


@EVALUATOR_REGISTRY.register("oriented_object_detection")
class OBBEvaluator(EvaluatorBase):
    """OBB (Oriented Bounding Box) Evaluator for DOTA dataset."""

    def __init__(self, session: SessionBase, dataset: DatasetBase, **kwargs) -> None:
        super().__init__(session, dataset, workers=12, **kwargs)
        self.iouv = np.linspace(0.5, 0.95, 10)
        self.niou = len(self.iouv)
        self.stats = []

    def init_metrics(self) -> dict:
        self.stats = []
        return {"processed_images": 0}

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
        if isinstance(origin_shape, (list, tuple)):
            origin_shape = [v[0] if hasattr(v, "__getitem__") else v for v in origin_shape]

        if output is None or (hasattr(output, "__len__") and len(output) == 0):
            pred = {"bboxes": np.zeros((0, 5)), "conf": np.zeros(0), "cls": np.zeros(0)}
        else:
            boxes = np.asarray(output["boxes"])
            scores = np.asarray(output["scores"])
            labels = np.asarray(output["labels"])
            angles = np.asarray(output["angles"])
            # bboxes in xywhr format: [cx, cy, w, h, angle]
            bboxes = np.concatenate([boxes[:, :4], angles.reshape(-1, 1)], axis=1)
            pred = {"bboxes": bboxes, "conf": scores, "cls": labels}

        gt_batch = self._prepare_batch(img_id, origin_shape)
        stat = self._process_batch(pred, gt_batch)
        self.stats.append(stat)
        metrics_state["processed_images"] += 1
        return metrics_state

    def compute_final_metrics(self, metrics_state: dict) -> dict:
        if not self.stats:
            return self._finalize(
                metric_names=["mAP", "mAP50"],
                metric_values=[0.0, 0.0],
                fps=0.0,
            )
        stats_dict = {}
        for k in self.stats[0].keys():
            stats_dict[k] = np.concatenate([s[k] for s in self.stats], 0)
        map50 = map_avg = 0.0
        if len(stats_dict["tp"]):
            results = self._ap_per_class(
                stats_dict["tp"],
                stats_dict["conf"],
                stats_dict["pred_cls"],
                stats_dict["target_cls"],
            )
            nt, fp, p, r, f1, ap, unique_classes = results
            if ap.size > 0:
                map50 = float(ap[:, 0].mean())
                map_avg = float(ap.mean())
        avg_fps = (
            metrics_state["processed_images"] / self.total_inference_time if self.total_inference_time > 0 else 0.0
        )
        return self._finalize(
            metric_names=["mAP", "mAP50"],
            metric_values=[map_avg * 100, map50 * 100],
            fps=avg_fps,
        )

    def format_progress_desc(self, metrics_state: dict, current_fps: float) -> str:
        img_count = metrics_state.get("processed_images", 0)
        return f"OBB | Images:{img_count} Current_FPS:{current_fps:.1f}"

    def _prepare_batch(self, img_id: Any, origin_shape: List) -> dict:
        img_id_str = str(img_id[0] if isinstance(img_id, (list, tuple)) else img_id)
        label_path = f"{self.dataset.label_dir}/{img_id_str}.txt"
        bboxes_list = []
        cls_list = []
        img_h = float(origin_shape[0].item() if hasattr(origin_shape[0], "item") else origin_shape[0])
        img_w = float(origin_shape[1].item() if hasattr(origin_shape[1], "item") else origin_shape[1])
        try:
            with open(label_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 9:
                        class_idx = int(parts[0])
                        coords = [float(p) for p in parts[1:9]]
                        x_coords = np.array(coords[0::2]) * img_w
                        y_coords = np.array(coords[1::2]) * img_h
                        points = np.array([[x_coords[i], y_coords[i]] for i in range(4)], dtype=np.float32)
                        (cx, cy), (w, h), angle_deg = cv2.minAreaRect(points)
                        angle = np.deg2rad(angle_deg)
                        if w < h:
                            w, h = h, w
                            angle += np.pi / 2
                        while angle >= 3 * np.pi / 4:
                            angle -= np.pi
                        while angle < -np.pi / 4:
                            angle += np.pi
                        bboxes_list.append([cx, cy, w, h, angle])
                        cls_list.append(class_idx)
        except FileNotFoundError:
            pass
        if not bboxes_list:
            return {"bboxes": np.zeros((0, 5)), "cls": np.zeros(0)}
        return {
            "bboxes": np.array(bboxes_list, dtype=np.float32),
            "cls": np.array(cls_list, dtype=np.float32),
        }

    def _process_batch(self, preds: dict, batch: dict) -> dict:
        if batch["cls"].shape[0] == 0 or preds["cls"].shape[0] == 0:
            return {
                "tp": np.zeros((preds["cls"].shape[0], self.niou), dtype=bool),
                "conf": preds["conf"],
                "pred_cls": preds["cls"],
                "target_cls": batch["cls"],
            }
        iou = batch_probiou(batch["bboxes"], preds["bboxes"])
        correct = self._match_predictions(preds["cls"], batch["cls"], iou)
        return {
            "tp": correct,
            "conf": preds["conf"],
            "pred_cls": preds["cls"],
            "target_cls": batch["cls"],
        }

    def _match_predictions(self, pred_cls: np.ndarray, true_cls: np.ndarray, iou: np.ndarray) -> np.ndarray:
        correct = np.zeros((pred_cls.shape[0], self.niou), dtype=bool)
        correct_class = true_cls[:, None] == pred_cls
        iou = iou * correct_class
        for i, threshold in enumerate(self.iouv):
            matches = np.nonzero(iou >= threshold)
            matches = np.array(matches).T
            if matches.shape[0]:
                if matches.shape[0] > 1:
                    matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True
        return correct

    def _ap_per_class(self, tp, conf, pred_cls, target_cls):
        i = np.argsort(-conf)
        tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
        unique_classes, nt = np.unique(target_cls, return_counts=True)
        nc = unique_classes.shape[0]
        ap = np.zeros((nc, tp.shape[1]))
        p = np.zeros(nc)
        r = np.zeros(nc)
        for ci, c in enumerate(unique_classes):
            i = pred_cls == c
            n_l = nt[ci]
            n_p = i.sum()
            if n_p == 0 or n_l == 0:
                continue
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)
            recall = tpc / (n_l + 1e-16)
            r[ci] = recall[:, 0].max() if recall.shape[0] else 0.0
            precision = tpc / (tpc + fpc)
            p[ci] = precision[:, 0].max() if precision.shape[0] else 0.0
            for j in range(tp.shape[1]):
                ap[ci, j] = self._compute_ap(recall[:, j], precision[:, j])
        f1 = 2 * p * r / (p + r + 1e-16)
        return (nt.astype(int), np.zeros(nc), p, r, f1, ap, unique_classes.astype(int))

    @staticmethod
    def _compute_ap(recall, precision):
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
        x = np.linspace(0, 1, 101)
        _trapz = getattr(np, "trapezoid", None) or np.trapz
        return float(_trapz(np.interp(x, mrec, mpre), x))
