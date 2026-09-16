import numpy as np
import pytest

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.evaluator.anomaly_detection_evaluator import AnomalyDetectionEvaluator
from dx_modelzoo.session import SessionBase


class FakeSession(SessionBase):
    def __init__(self, path: str):
        super().__init__(path)

    def run(self, inputs):
        return [np.asarray(inputs)]


class FakeDataset(DatasetBase):
    categories = ["bottle"]

    def __init__(self):
        super().__init__("/fake")

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise IndexError(idx)

    def validate(self):
        pass


def _build_evaluator(method: str):
    evaluator = AnomalyDetectionEvaluator(FakeSession("/compiled/model.dxnn"), FakeDataset())
    evaluator.method = method
    return evaluator


def test_efficientad_component_prepares_independent_feature_banks(monkeypatch):
    evaluator = _build_evaluator("efficientad_component")
    calls = []

    def build_bank(category):
        calls.append(category)
        return object()

    monkeypatch.setattr(evaluator, "_build_bank", build_bank)

    evaluator.init_metrics()

    assert calls == ["bottle"]
    assert list(evaluator._banks) == ["bottle"]
    assert evaluator._banks["bottle"] is not None


def test_efficientad_requires_sibling_bundle():
    evaluator = _build_evaluator("efficientad")

    with pytest.raises(FileNotFoundError, match="requires teacher, student, and autoencoder"):
        evaluator._ea_setup_sessions()
