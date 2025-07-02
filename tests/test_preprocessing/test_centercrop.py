import numpy as np
import torch
from torchvision.transforms.v2.functional import center_crop

from dx_modelzoo.preprocessing.centercrop import CenterCrop


def test_centercrop():
    width = 224
    height = 224
    centercrop = CenterCrop(width, height)

    dummy_input = torch.randn(256, 256, 3)
    torch_input = dummy_input.permute(2, 0, 1).unsqueeze(0)

    center_crop_output = centercrop(dummy_input.numpy())
    answer = center_crop(torch_input, [height, width])
    answer = answer[0].permute(1, 2, 0).numpy()

    assert center_crop_output.shape == answer.shape
    assert np.allclose(center_crop_output, answer)
