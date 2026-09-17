"""Tests for class-agnostic mainbody top-1 localization."""

import numpy as np

from dx_modelzoo.evaluator.mainbody_detection_evaluator import MainbodyDetectionEvaluator, top1_hit


def test_top1_hit_matches_any_ground_truth_box():
    predictions = np.array([[0, 0, 10, 10, 0.9, 0]], dtype=np.float32)
    ground_truth = np.array([[0, 0, 10, 10], [40, 40, 50, 50]], dtype=np.float32)

    assert top1_hit(predictions, ground_truth, iou_threshold=0.5)


def test_top1_hit_uses_highest_confidence_prediction():
    predictions = np.array(
        [
            [40, 40, 50, 50, 0.9, 0],
            [0, 0, 10, 10, 0.1, 0],
        ],
        dtype=np.float32,
    )
    ground_truth = np.array([[0, 0, 10, 10]], dtype=np.float32)

    assert not top1_hit(predictions, ground_truth, iou_threshold=0.5)


def test_final_metric_is_recall_at_one():
    evaluator = object.__new__(MainbodyDetectionEvaluator)
    evaluator.total_inference_time = 1.0
    evaluator._start_time = None
    evaluator.model_name = "test"
    evaluator.model_spec = {}
    evaluator.dataset_name = "COCO"
    evaluator.profile_name = "onnx"
    evaluator._finalize = lambda metric_names, metric_values, fps: {
        "metrics": [{"name": metric_names[0], "metric_value": metric_values[0]}]
    }

    result = evaluator.compute_final_metrics({"hits": 3, "images": 4})

    assert result["metrics"] == [{"name": "Recall@1", "metric_value": 75.0}]
