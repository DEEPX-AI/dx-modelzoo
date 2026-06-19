"""Tests for dx_modelzoo.evaluator.person_attribute_evaluator."""

import numpy as np
from datetime import datetime

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.evaluator.person_attribute_evaluator import PersonAttributeEvaluator
from dx_modelzoo.session import SessionBase


class FakeSession(SessionBase):
    def __init__(self):
        super().__init__("/fake.onnx")

    def run(self, inputs):
        return [inputs]


class FakeDataset(DatasetBase):
    def __init__(self):
        super().__init__("/fake")
        self.num_attr = 35

    def __len__(self):
        return 3

    def __getitem__(self, idx):
        img = np.zeros((3, 256, 128), dtype=np.float32)
        labels = np.zeros(35, dtype=np.int64)
        return img, labels, idx


class TestPersonAttributeEvaluator:
    def _make(self):
        ev = PersonAttributeEvaluator(FakeSession(), FakeDataset())
        ev._start_time = datetime.now()
        return ev

    def test_init_metrics(self):
        ev = self._make()
        metrics = ev.init_metrics()
        assert "tp" in metrics

    def test_process_and_compute(self):
        ev = self._make()
        ev.total_inference_time = 1.0
        metrics = ev.init_metrics()
        # Binary multi-label: logits (1, 5), sigmoid applied internally
        batch_data = (np.zeros((3, 256, 128)), np.zeros(35, dtype=np.int64), 0)
        output = [np.array(np.random.rand(1, 35).astype(np.float32), dtype=np.float32)]
        metrics = ev.process_batch_result(batch_data, output, metrics)
        assert metrics["count"] == 1
        result = ev.compute_final_metrics(metrics)
        assert isinstance(result, dict)
