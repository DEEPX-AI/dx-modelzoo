import numpy as np
import torch

from dx_modelzoo.preprocessing.div import Div


def test_div():
    x = 3

    div = Div(x)

    dummy_input = torch.randn(224, 224, 3).numpy()

    output = div(dummy_input)
    answer = dummy_input / x

    assert np.allclose(output, answer)
