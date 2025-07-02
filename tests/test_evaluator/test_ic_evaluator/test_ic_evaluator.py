import numpy as np
import torch

from dx_modelzoo.evaluator.ic_evaluator import ICEvaluator
from dx_modelzoo.models.image_classification import topk_postprocessing


def test_ic_evaluator(dummy_ic_dataset, dummy_session):
    evaluator = ICEvaluator(dummy_session, dummy_ic_dataset)
    evaluator.set_postprocessing(topk_postprocessing)

    evaluator.eval()


def test_run_one_batch(dummy_ic_dataset, dummy_session):
    evaluator = ICEvaluator(dummy_session, dummy_ic_dataset)
    evaluator.set_postprocessing(topk_postprocessing)

    out = evaluator._run_one_batch(torch.randn(1, 3, 224, 224), torch.tensor([1]))
    assert out.shape == (1, 5)


def test_topk_eval(dummy_ic_dataset, dummy_session):
    evaluator = ICEvaluator(dummy_session, dummy_ic_dataset)
    evaluator.set_postprocessing(topk_postprocessing)

    topk_correct_count = [0, 0]

    output = evaluator._topk_eval(topk_correct_count, np.array([[False, False, False, True, False]]))
    assert output[0] == 0
    assert output[1] == 1
