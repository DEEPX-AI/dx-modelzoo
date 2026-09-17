"""Tests for person ReID metric units."""

import numpy as np

from dx_modelzoo.evaluator.person_reid_evaluator import PersonReIDEvaluator


class FakeDataset:
    gallery = [0, 1]
    query = [0, 1]
    gallery_ids = np.array([1, 2])
    gallery_cams = np.array([2, 2])
    query_ids = np.array([1, 2])
    query_cams = np.array([1, 1])


def test_person_reid_metrics_are_reported_as_percentages():
    evaluator = object.__new__(PersonReIDEvaluator)
    evaluator._gallery_embeddings = [np.eye(2, dtype=np.float32)]
    evaluator.dataset = FakeDataset()
    evaluator.total_inference_time = 1.0
    evaluator._finalize = lambda metric_names, metric_values, fps: {
        "names": metric_names,
        "values": metric_values,
    }

    result = evaluator.compute_final_metrics({"embeddings": [np.eye(2, dtype=np.float32)]})

    assert result["names"] == ["Rank-1", "mAP"]
    assert result["values"] == [100.0, 100.0]
