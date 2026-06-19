"""Tests for dx_modelzoo.evaluator.panoptic_driving_perception_evaluator."""

import numpy as np

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.evaluator.panoptic_driving_perception_evaluator import PanopticDrivingPerceptionEvaluator
from dx_modelzoo.session import SessionBase


class FakeSession(SessionBase):
    def __init__(self):
        super().__init__("/fake.onnx")

    def run(self, inputs):
        return [inputs]


class FakeDataset(DatasetBase):
    def __init__(self):
        super().__init__("/fake")
        # Common attrs evaluators may expect
        self.num_class = 10
        self.num_classes = 10

    def __len__(self):
        return 3

    def __getitem__(self, idx):
        return np.zeros((3, 64, 64), dtype=np.float32), idx


class TestPanopticDrivingPerceptionEvaluator:
    def _make(self):
        return PanopticDrivingPerceptionEvaluator(FakeSession(), FakeDataset())

    def test_init_metrics(self):
        ev = self._make()
        metrics = ev.init_metrics()
        assert isinstance(metrics, dict)

    def test_compute_final_metrics(self):
        ev = self._make()
        ev.total_inference_time = 1.0
        metrics = ev.init_metrics()
        # Set minimal count to avoid division by zero
        if "current_count" in metrics:
            metrics["current_count"] = 1
        if "count" in metrics:
            metrics["count"] = 1
        if "image_count" in metrics:
            metrics["image_count"] = 1
        if "sample_count" in metrics:
            metrics["sample_count"] = 1
        if "total_samples" in metrics:
            metrics["total_samples"] = 1
        result = ev.compute_final_metrics(metrics)
        assert isinstance(result, dict)
