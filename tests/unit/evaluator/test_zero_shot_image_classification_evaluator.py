"""Tests for dx_modelzoo.evaluator.zero_shot_image_classification_evaluator (E2E)."""

import numpy as np

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.evaluator.zero_shot_image_classification_evaluator import ZeroShotImageClassificationEvaluator
from dx_modelzoo.postprocessing.topk import TopK
from dx_modelzoo.session import SessionBase


class FakeSession(SessionBase):
    def __init__(self, path="/fake.onnx"):
        super().__init__(path)

    def run(self, inputs):
        logits = np.random.rand(1, 10).astype(np.float32) * 0.1
        logits[0, 0] = 1.0  # Always predict class 0
        return [logits]


class FakeDataset(DatasetBase):
    def validate(self) -> None:  # ponytail: test double, nothing to validate
        pass

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

    def test_discovers_text_embedding_cache_before_building(self, tmp_path):
        model_path = tmp_path / "clip-image.onnx"
        model_path.touch()
        cached = np.ones((4, 3), dtype=np.float32)
        np.save(tmp_path / "imagenet_text_embedding.npy", cached)

        ev = ZeroShotImageClassificationEvaluator(FakeSession(str(model_path)), FakeDataset())
        ev.clip_model_name = "not-needed-with-cache"
        ev.clip_pretrained = "not-needed-with-cache"

        ev.init_metrics()

        np.testing.assert_array_equal(ev.zeroshot_text_embedding_weight, cached)

    def test_eval_end_to_end(self):
        ev = self._make()
        result = ev.eval()
        assert isinstance(result, dict)
