"""Tests for dx_modelzoo.evaluator.super_resolution_evaluator (E2E)."""

import numpy as np

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.evaluator.super_resolution_evaluator import SuperResolutionEvaluator
from dx_modelzoo.session import SessionBase


class FakeSession(SessionBase):
    def __init__(self):
        super().__init__("/fake.onnx")

    def run(self, inputs):
        # SR output: same as input (identity)
        return [inputs]


class FakeDataset(DatasetBase):
    def __init__(self):
        super().__init__("/fake")
        self.scale = 2

    def __len__(self):
        return 3

    def __getitem__(self, idx):
        lr = np.random.rand(3, 16, 16).astype(np.float32)
        hr = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
        return lr, hr


class TestSuperResolutionEvaluator:
    def _make(self):
        ev = SuperResolutionEvaluator(FakeSession(), FakeDataset())
        ev.set_postprocessing(lambda outputs, **kw: outputs[0] if isinstance(outputs, list) else outputs)
        return ev

    def test_init_metrics(self):
        ev = self._make()
        metrics = ev.init_metrics()
        assert metrics["total_psnr"] == 0.0

    def test_eval_end_to_end(self):
        ev = self._make()
        result = ev.eval()
        assert isinstance(result, dict)
