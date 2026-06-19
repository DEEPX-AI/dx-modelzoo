"""Tests for dx_modelzoo.evaluator.image_denoising_evaluator (E2E)."""

import numpy as np

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.evaluator.image_denoising_evaluator import ImageDenoisingEvaluator
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
        noisy = np.random.rand(1, 1, 64, 64).astype(np.float32)
        clean = (np.random.rand(64, 64) * 255).astype(np.uint8)
        return noisy, clean


class TestImageDenoisingEvaluator:
    def _make(self):
        ev = ImageDenoisingEvaluator(FakeSession(), FakeDataset())
        ev.set_postprocessing(lambda outputs, **kw: outputs[0] if isinstance(outputs, list) else outputs)
        return ev

    def test_init_metrics(self):
        ev = self._make()
        metrics = ev.init_metrics()
        assert "psnr_list" in metrics

    def test_eval_end_to_end(self):
        ev = self._make()
        result = ev.eval()
        assert isinstance(result, dict)
