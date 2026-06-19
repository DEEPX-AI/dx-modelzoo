"""Tests for dx_modelzoo.evaluator.semantic_segmentation_evaluator."""

import numpy as np
from datetime import datetime

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.evaluator.semantic_segmentation_evaluator import SegmentationEvaluator
from dx_modelzoo.session import SessionBase


class FakeSession(SessionBase):
    def __init__(self, num_classes=3):
        super().__init__("/fake.onnx")
        self.num_classes = num_classes

    def run(self, inputs):
        # Return NCHW logits where class 1 always wins
        h, w = inputs.shape[-2], inputs.shape[-1]
        out = np.zeros((1, self.num_classes, h, w), dtype=np.float32)
        out[0, 1, :, :] = 1.0
        return [out]


class FakeDataset(DatasetBase):
    def __init__(self, num_classes=3):
        super().__init__("/fake")
        self.num_classes = num_classes
        self.num_class = num_classes

    def __len__(self):
        return 3

    def __getitem__(self, idx):
        img = np.zeros((3, 32, 32), dtype=np.float32)
        label = np.ones((32, 32), dtype=np.int64)  # All class 1
        return img, label


class TestSemanticSegmentationEvaluator:
    def _make_evaluator(self, num_classes=3):
        session = FakeSession(num_classes)
        dataset = FakeDataset(num_classes)
        ev = SegmentationEvaluator(session, dataset)
        from dx_modelzoo.postprocessing.segmentation_argmax import SegmentationArgmax
        ev.set_postprocessing(SegmentationArgmax(layout="nchw"))
        return ev

    def test_init_metrics(self):
        ev = self._make_evaluator()
        metrics = ev.init_metrics()
        assert "confusion_matrix" in metrics
        assert metrics["confusion_matrix"].shape == (3, 3)

    def test_calculate_miou_perfect(self):
        ev = self._make_evaluator(num_classes=3)
        cm = np.eye(3) * 10
        miou = ev.calculate_miou(cm)
        np.testing.assert_almost_equal(miou, 1.0)

    def test_calculate_miou_zero(self):
        ev = self._make_evaluator(num_classes=3)
        cm = np.zeros((3, 3))
        miou = ev.calculate_miou(cm)
        assert miou == 0.0 or np.isnan(miou)

    def test_eval_end_to_end(self):
        ev = self._make_evaluator(num_classes=3)
        result = ev.eval()
        assert isinstance(result, dict)
        # Prediction is always class 1, label is always class 1 → perfect mIoU for class 1
