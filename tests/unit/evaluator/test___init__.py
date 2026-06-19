"""Tests for dx_modelzoo.evaluator base (EvaluatorBase + registry)."""

import numpy as np
import pytest

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.evaluator import EVALUATOR_REGISTRY, EvaluatorBase
from dx_modelzoo.session import SessionBase


class FakeSession(SessionBase):
    """Minimal session mock for testing."""

    def __init__(self):
        super().__init__("/fake/model.onnx")

    def run(self, inputs):
        return [inputs]


class FakeDataset(DatasetBase):
    def __init__(self, size=5):
        super().__init__("/fake")
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return np.zeros((3, 32, 32), dtype=np.float32), idx


class TestEvaluatorRegistry:
    def test_image_classification_registered(self):
        assert "image_classification" in EVALUATOR_REGISTRY

    def test_object_detection_registered(self):
        assert "object_detection" in EVALUATOR_REGISTRY

    def test_semantic_segmentation_registered(self):
        assert "semantic_segmentation" in EVALUATOR_REGISTRY

    def test_registry_has_many_evaluators(self):
        assert len(EVALUATOR_REGISTRY) >= 20


class TestEvaluatorBase:
    def test_init(self):
        session = FakeSession()
        dataset = FakeDataset()
        # Use a concrete evaluator from registry
        EvalCls = EVALUATOR_REGISTRY.get("image_classification")
        evaluator = EvalCls(session, dataset)
        assert evaluator.session is session
        assert evaluator.dataset is dataset

    def test_postprocessing_not_set_raises(self):
        session = FakeSession()
        dataset = FakeDataset()
        EvalCls = EVALUATOR_REGISTRY.get("image_classification")
        evaluator = EvalCls(session, dataset)
        with pytest.raises(ValueError, match="not set"):
            _ = evaluator.postprocessing

    def test_set_postprocessing(self):
        session = FakeSession()
        dataset = FakeDataset()
        EvalCls = EVALUATOR_REGISTRY.get("image_classification")
        evaluator = EvalCls(session, dataset)
        evaluator.set_postprocessing(lambda x, **kw: x)
        assert evaluator.postprocessing is not None
