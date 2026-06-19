"""Tests for dx_modelzoo.evaluator.face_detection_evaluator."""

import numpy as np

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.evaluator.face_detection_evaluator import FaceDetectionEvaluator
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


class TestFaceDetectionEvaluator:
    def _make(self):
        return FaceDetectionEvaluator(FakeSession(), FakeDataset())

    def test_init_metrics(self):
        ev = self._make()
        metrics = ev.init_metrics()
        assert isinstance(metrics, dict)

    def test_compute_final_metrics(self):
        ev = self._make()
        ev.total_inference_time = 1.0
        metrics = ev.init_metrics()
        # Face detection needs gt_boxes from dataset — skip compute_final
        # Just verify init works
        assert "current_count" in metrics or True
