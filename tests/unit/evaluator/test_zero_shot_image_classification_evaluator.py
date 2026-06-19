"""Tests for dx_modelzoo.evaluator.zero_shot_image_classification_evaluator (E2E)."""

import numpy as np
from datetime import datetime

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.evaluator.zero_shot_image_classification_evaluator import ZeroShotImageClassificationEvaluator
from dx_modelzoo.postprocessing.topk import TopK
from dx_modelzoo.session import SessionBase


class FakeSession(SessionBase):
    def __init__(self):
        super().__init__("/fake.onnx")

    def run(self, inputs):
        logits = np.random.rand(1, 10).astype(np.float32) * 0.1
        logits[0, 0] = 1.0  # Always predict class 0
        return [logits]


class FakeDataset(DatasetBase):
    def __init__(self):
        super().__init__("/fake")

    def __len__(self):
        return 3

    def __getitem__(self, idx):
        return np.zeros((3, 224, 224), dtype=np.float32), 0


class TestZeroShotImageClassificationEvaluator:
    def _make(self):
        ev = ZeroShotImageClassificationEvaluator(FakeSession(), FakeDataset())
        ev.set_postprocessing(TopK(k=[1, 5]))
        return ev

    def test_init_metrics(self):
        ev = self._make()
        metrics = ev.init_metrics()
        assert "correct_top1" in metrics

    def test_eval_end_to_end(self):
        ev = self._make()
        result = ev.eval()
        assert isinstance(result, dict)
