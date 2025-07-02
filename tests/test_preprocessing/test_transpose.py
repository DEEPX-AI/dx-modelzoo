import numpy as np
import torch

from dx_modelzoo.preprocessing.transpose import Transpose


def test_transpose():
    axis = [2, 0, 1]
    transpose = Transpose(axis)

    dummy_input = torch.randn(224, 224, 3).numpy()

    output = transpose(dummy_input)
    answer = np.transpose(dummy_input, axes=axis)

    assert output.shape == answer.shape
    assert np.allclose(output, answer)
