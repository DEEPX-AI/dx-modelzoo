import cv2
import numpy as np
import pytest
import torch
from torchvision.transforms.v2.functional import resize as torch_resize

from dx_modelzoo.preprocessing.resize import INTERPOLATION_MAP, Resize


@pytest.mark.parametrize("mode, size, interpolation", [("cv2", [256, 256], "LINEAR"), ("torchvision", 256, "BILINEAR")])
def test_resize(mode, size, interpolation):

    dummy_input = torch.randn(640, 640, 3)
    resize = Resize(mode, size, interpolation)

    output = resize(dummy_input.numpy())

    if mode == "cv2":
        answer = cv2.resize(dummy_input.numpy(), size, INTERPOLATION_MAP[interpolation])
    elif mode == "torchvision":
        dummy_input = dummy_input.permute(2, 0, 1).unsqueeze(0)
        answer = torch_resize(dummy_input, size, INTERPOLATION_MAP[interpolation])
        answer = answer.squeeze(0).permute(1, 2, 0).numpy()

    assert output.shape == answer.shape
    assert np.allclose(output, answer)
