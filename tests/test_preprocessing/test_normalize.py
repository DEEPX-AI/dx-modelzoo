import numpy as np
import pytest
import torch
from torchvision.transforms.v2.functional import normalize as torch_normalize

from dx_modelzoo.preprocessing.normalize import Normalize


@pytest.mark.parametrize("permute", [True, False])
def test_normalize(permute):
    mean = [2, 2, 2]
    std = [3, 3, 3]
    normalize = Normalize(mean, std)

    if permute:
        dummy_input = torch.randn(224, 224, 3)
        torch_input = dummy_input.permute(2, 0, 1)

        output = normalize(dummy_input.clone().numpy())
        answer = torch_normalize(torch_input, mean, std)
        answer = answer.permute(1, 2, 0).numpy()
    else:
        dummy_input = torch.randn(3, 224, 224)
        torch_input = dummy_input

        output = normalize(dummy_input.clone().numpy())
        answer = torch_normalize(torch_input, mean, std)
        answer = answer.numpy()

    assert np.allclose(output, answer)
