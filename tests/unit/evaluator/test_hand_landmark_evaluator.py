"""Tests for dx_modelzoo.evaluator.hand_landmark_evaluator."""

import numpy as np
from datetime import datetime

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.evaluator.hand_landmark_evaluator import HandLandmarkEvaluator
from dx_modelzoo.session import SessionBase


class FakeSession(SessionBase):
    def __init__(self):
        super().__init__("/fake.onnx")

    def run(self, inputs):
        # Return 21 keypoints × 3 (x, y, visibility)
        return [np.random.rand(1, 63).astype(np.float32) * 192]


class FakeDataset(DatasetBase):
    def __init__(self):
        super().__init__("/fake")

    def __len__(self):
        return 3

    def __getitem__(self, idx):
        img = np.zeros((3, 192, 192), dtype=np.float32)
        # GT keypoints: 21 × 3 (normalized 0-1)
        gt_kpts = np.random.rand(21, 3).astype(np.float32)
        return img, gt_kpts, idx


class TestHandLandmarkEvaluator:
    def _make(self):
        ev = HandLandmarkEvaluator(FakeSession(), FakeDataset())
        ev._start_time = datetime.now()
        return ev

    def test_init_metrics(self):
        ev = self._make()
        metrics = ev.init_metrics()
        assert "total_mnae" in metrics
        assert "count" in metrics

    def test_process_and_compute(self):
        ev = self._make()
        ev.total_inference_time = 1.0
        metrics = ev.init_metrics()
        img = np.zeros((3, 192, 192), dtype=np.float32)
        gt_kpts = np.ones((21, 3), dtype=np.float32) * 0.5
        batch_data = (img, gt_kpts, 0)
        output = [np.ones((1, 63), dtype=np.float32) * 96]  # pixel coords → 96/192 = 0.5
        metrics = ev.process_batch_result(batch_data, output, metrics)
        assert metrics["count"] == 1
        result = ev.compute_final_metrics(metrics)
        assert isinstance(result, dict)
