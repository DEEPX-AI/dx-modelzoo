"""Tests for dx_modelzoo.evaluator.face_recognition_evaluator."""

import numpy as np

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.evaluator.face_recognition_evaluator import FaceRecognitionEvaluator
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


class TestFaceRecognitionEvaluator:
    def _make(self):
        return FaceRecognitionEvaluator(FakeSession(), FakeDataset())

    def test_init_metrics(self):
        ev = self._make()
        metrics = ev.init_metrics()
        assert isinstance(metrics, dict)
        assert "embeddings1" in metrics
        assert "embeddings2" in metrics

    def test_compute_final_metrics(self):
        ev = self._make()
        ev.total_inference_time = 1.0
        metrics = ev.init_metrics()
        # Need real embeddings to compute
        metrics["embeddings1"] = [np.random.rand(1, 128).astype(np.float32)]
        metrics["embeddings2"] = [np.random.rand(1, 128).astype(np.float32)]
        metrics["labels"] = [1]
        result = ev.compute_final_metrics(metrics)
        assert isinstance(result, dict)
