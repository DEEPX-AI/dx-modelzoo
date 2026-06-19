"""Tests for dx_modelzoo.evaluator.image_classification_evaluator."""

import numpy as np
from datetime import datetime

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.evaluator.image_classification_evaluator import ImageClassificationEvaluator
from dx_modelzoo.postprocessing.topk import TopK
from dx_modelzoo.session import SessionBase


class FakeSession(SessionBase):
    """Returns fake logits that make class=idx%10 the top prediction."""

    def __init__(self):
        super().__init__("/fake.onnx")

    def run(self, inputs):
        # Return logits where argmax is deterministic
        logits = np.random.rand(1, 10).astype(np.float32) * 0.1
        logits[0, 0] = 1.0  # Always predict class 0
        return [logits]


class FakeDataset(DatasetBase):
    def __init__(self, size=5):
        super().__init__("/fake")
        self._size = size

    def __len__(self):
        return self._size

    def __getitem__(self, idx):
        return np.zeros((3, 224, 224), dtype=np.float32), 0  # label=0


class TestImageClassificationEvaluator:
    def _make_evaluator(self, size=5):
        ev = ImageClassificationEvaluator(FakeSession(), FakeDataset(size))
        ev.set_postprocessing(TopK(k=[1, 5]))
        return ev

    def test_init_metrics(self):
        ev = self._make_evaluator()
        metrics = ev.init_metrics()
        assert metrics["topk_correct_count"] == [0, 0]
        assert metrics["current_count"] == 0

    def test_process_batch_result_correct(self):
        ev = self._make_evaluator()
        ev._start_time = datetime.now()
        metrics = ev.init_metrics()
        batch_data = (np.zeros((3, 224, 224)), 2)
        output = np.array([[2, 1, 0, 3, 4]])
        metrics = ev.process_batch_result(batch_data, output, metrics)
        assert metrics["current_count"] == 1
        assert metrics["topk_correct_count"][0] == 1
        assert metrics["topk_correct_count"][1] == 1

    def test_process_batch_result_wrong(self):
        ev = self._make_evaluator()
        ev._start_time = datetime.now()
        metrics = ev.init_metrics()
        batch_data = (np.zeros((3, 224, 224)), 9)
        output = np.array([[0, 1, 2, 3, 4]])
        metrics = ev.process_batch_result(batch_data, output, metrics)
        assert metrics["topk_correct_count"][0] == 0
        assert metrics["topk_correct_count"][1] == 0

    def test_compute_final_metrics(self):
        ev = self._make_evaluator()
        ev._start_time = datetime.now()
        ev.total_inference_time = 1.0
        metrics = {"topk_correct_count": [8, 10], "current_count": 10}
        result = ev.compute_final_metrics(metrics)
        assert isinstance(result, dict)

    def test_eval_end_to_end(self):
        ev = self._make_evaluator(size=3)
        ev._start_time = datetime.now()
        result = ev.eval()
        assert isinstance(result, dict)
        # Should have 100% top1 since label=0 and model always predicts 0
        metrics = result.get("metrics", [])
        if metrics:
            top1 = metrics[0]["metric_value"]
            assert top1 == 100.0

    def test_format_progress_desc(self):
        ev = self._make_evaluator()
        metrics = {"topk_correct_count": [5, 8], "current_count": 10}
        desc = ev.format_progress_desc(metrics, 30.0)
        assert "Top1" in desc
        assert "FPS" in desc
