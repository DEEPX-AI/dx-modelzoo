"""Tests for dx_modelzoo.evaluator.low_light_enhancement_evaluator (E2E)."""

import numpy as np
from datetime import datetime

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.evaluator.low_light_enhancement_evaluator import LowLightEnhancementEvaluator
from dx_modelzoo.session import SessionBase


class FakeSession(SessionBase):
    def __init__(self):
        super().__init__("/fake.onnx")

    def run(self, inputs):
        h, w = inputs.shape[-2], inputs.shape[-1]
        return [np.random.rand(1, 3, h, w).astype(np.float32)]


class FakeDataset(DatasetBase):
    def __init__(self):
        super().__init__("/fake")

    def __len__(self):
        return 3

    def __getitem__(self, idx):
        return np.zeros((3, 64, 64), dtype=np.float32), (np.random.rand(64, 64, 3) * 255).astype(np.uint8)


class TestLowLightEnhancementEvaluator:
    def _make(self):
        ev = LowLightEnhancementEvaluator(FakeSession(), FakeDataset())
        ev.set_postprocessing(lambda outputs, **kw: outputs[0] if isinstance(outputs, list) else outputs)
        return ev

    def test_init_metrics(self):
        ev = self._make()
        metrics = ev.init_metrics()
        assert isinstance(metrics, dict)

    def test_eval_end_to_end(self):
        ev = self._make()
        result = ev.eval()
        assert isinstance(result, dict)
