"""Tests for dx_modelzoo.evaluator.object_detection_3d_evaluator."""

import numpy as np

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.evaluator.object_detection_3d_evaluator import ObjectDetection3DEvaluator, _rotated_iou
from dx_modelzoo.session import SessionBase


class FakeSession(SessionBase):
    def __init__(self):
        super().__init__("/fake.onnx")

    def run(self, inputs):
        return [inputs]


class FakeDataset(DatasetBase):
    """Single GT car at LiDAR (x=20, y=5), l=4, w=2, yaw=0."""

    def __init__(self):
        super().__init__("/fake")

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return np.zeros((1, 3, 608, 608), dtype=np.float32), (608, 608, 3), "000000"

    def get_gt(self, sample_id):
        boxes = np.array([[20.0, 5.0, -1.0, 1.5, 2.0, 4.0, 0.0]])  # x,y,z,h,w,l,yaw
        return boxes, np.array([1.0])


def _pred_dict(x, y, length, w, yaw, score, cls):
    """Build a decode-style dict for one BEV detection (real LiDAR coords)."""
    from dx_modelzoo.dataset.kitti import BEV_HEIGHT, BEV_WIDTH, BOUND_SIZE_X, BOUND_SIZE_Y, BOUNDARY

    cy = (x - BOUNDARY["minX"]) / BOUND_SIZE_X * BEV_HEIGHT
    cx = (y - BOUNDARY["minY"]) / BOUND_SIZE_Y * BEV_WIDTH
    boxes = np.array([[cx - 1, cy - 1, cx + 1, cy + 1]])  # pseudo xyxy, center recovered by evaluator
    extra = np.array([[0.0, 1.5, w, length, -yaw]])  # z,h,w,l,yaw_model (yaw_lidar = -yaw_model)
    return {
        "boxes": boxes,
        "scores": np.array([score]),
        "class_ids": np.array([float(cls)]),
        "extra": extra,
    }


def test_rotated_iou_edges():
    a = np.array([[0.0, 0.0, 4.0, 2.0, 0.0]])
    assert abs(_rotated_iou(a, a)[0, 0] - 1.0) < 1e-6
    far = np.array([[100.0, 100.0, 4.0, 2.0, 0.0]])
    assert _rotated_iou(a, far)[0, 0] == 0.0
    half = np.array([[2.0, 0.0, 4.0, 2.0, 0.0]])
    assert abs(_rotated_iou(a, half)[0, 0] - 1.0 / 3.0) < 1e-3


def test_perfect_prediction_scores_full_ap():
    ev = ObjectDetection3DEvaluator(FakeSession(), FakeDataset())
    ms = ev.init_metrics()
    out = _pred_dict(20.0, 5.0, 4.0, 2.0, 0.0, score=0.9, cls=1)
    ms = ev.process_batch_result((None, None, "000000"), out, ms)
    ev.total_inference_time = 1.0
    res = ev.compute_final_metrics(ms)
    values = {m["name"]: m["metric_value"] for m in res["metrics"]}
    assert values["mAP_BEV@0.5"] > 90.0
    assert values["mAP_BEV@0.7"] > 90.0


def test_far_prediction_scores_zero_ap():
    ev = ObjectDetection3DEvaluator(FakeSession(), FakeDataset())
    ms = ev.init_metrics()
    out = _pred_dict(45.0, -20.0, 4.0, 2.0, 0.0, score=0.9, cls=1)
    ms = ev.process_batch_result((None, None, "000000"), out, ms)
    ev.total_inference_time = 1.0
    res = ev.compute_final_metrics(ms)
    values = {m["name"]: m["metric_value"] for m in res["metrics"]}
    assert values["mAP_BEV@0.5"] == 0.0
