import numpy as np
import pytest

from dx_modelzoo.evaluator.image_matting_evaluator import conn, grad, mse, sad


@pytest.fixture
def alpha():
    a = np.zeros((64, 64), dtype=np.float32)
    a[16:48, 16:48] = 1.0
    a[16:48, 12:16] = 0.5
    return a


def test_perfect_prediction_scores_zero(alpha):
    assert sad(alpha, alpha) == 0.0
    assert mse(alpha, alpha) == 0.0
    assert grad(alpha, alpha) == pytest.approx(0.0, abs=1e-6)
    assert conn(alpha, alpha) == pytest.approx(0.0, abs=1e-6)


def test_sad_and_mse_match_closed_form(alpha):
    pred = np.clip(alpha + 0.25, 0.0, 1.0)
    diff = np.abs(pred - alpha)
    assert sad(pred, alpha) == pytest.approx(diff.sum() / 1000.0)
    assert mse(pred, alpha) == pytest.approx((diff**2).sum() / alpha.size)


def test_errors_grow_with_deviation(alpha):
    near = np.clip(alpha + 0.1, 0.0, 1.0)
    far = np.clip(alpha + 0.4, 0.0, 1.0)
    assert sad(near, alpha) < sad(far, alpha)
    assert mse(near, alpha) < mse(far, alpha)


def test_grad_penalises_blurred_edges(alpha):
    import cv2

    blurred = cv2.GaussianBlur(alpha, (9, 9), 3)
    assert grad(blurred, alpha) > grad(alpha, alpha)
