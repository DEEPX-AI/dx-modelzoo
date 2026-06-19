"""Tests for dx_modelzoo.evaluator.face_attribute_evaluator."""

import numpy as np
from datetime import datetime

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.evaluator.face_attribute_evaluator import FaceAttributeEvaluator
from dx_modelzoo.session import SessionBase


class FakeSession(SessionBase):
    def __init__(self):
        super().__init__("/fake.onnx")

    def run(self, inputs):
        return [inputs]


class FakeDataset(DatasetBase):
    def __init__(self):
        super().__init__("/fake")

    def __len__(self):
        return 3

    def __getitem__(self, idx):
        img = np.zeros((3, 218, 178), dtype=np.float32)
        labels = np.zeros(40, dtype=np.int64)
        return img, labels, idx


class TestFaceAttributeEvaluator:
    def _make(self):
        ev = FaceAttributeEvaluator(FakeSession(), FakeDataset())
        ev._start_time = datetime.now()
        return ev

    def test_init_metrics(self):
        ev = self._make()
        metrics = ev.init_metrics()
        assert metrics["correct"].shape == (40,)

    def test_process_and_compute(self):
        ev = self._make()
        ev.total_inference_time = 1.0
        metrics = ev.init_metrics()
        # logits: (40, 2) per attribute binary classification
        batch_data = (np.zeros((3, 218, 178)), np.zeros(40, dtype=np.int64), 0)
        output = [np.zeros((40, 2), dtype=np.float32)]
        output[0][:, 0] = 1.0  # predict class 0 for all attrs
        metrics = ev.process_batch_result(batch_data, output, metrics)
        assert metrics["count"] == 1
        result = ev.compute_final_metrics(metrics)
        assert isinstance(result, dict)
