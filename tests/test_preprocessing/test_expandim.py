import numpy as np
import torch

from dx_modelzoo.preprocessing.expanddim import ExpandDim


def test_expanddim():
    axis = 0
    expand_dim = ExpandDim(axis)

    dummy_input = torch.randn(3, 224, 224).numpy()

    output = expand_dim(dummy_input)
    answer = np.expand_dims(dummy_input, axis)

    assert output.shape == answer.shape
    assert np.allclose(output, answer)
