from typing import Callable

import cv2
import numpy as np
import pytest
import torch
from torchvision.transforms.v2.functional import center_crop, normalize, resize


def make_torch_input(inputs: np.ndarray) -> torch.Tensor:
    torch_inputs = torch.from_numpy(inputs)
    torch_inputs = torch_inputs.permute(2, 0, 1).unsqueeze(0)
    return torch_inputs


@pytest.fixture
def preprocessings1():
    return [
        {"resize": {"mode": "torchvision", "size": 256, "interpolation": "BILINEAR"}},
        {"centercrop": {"width": 224, "height": 224}},
        {"convertColor": {"form": "BGR2RGB"}},
        {"div": {"x": 255}},
        {"normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}},
        {"transpose": {"axis": [2, 0, 1]}},
        {"expandDim": {"axis": 0}},
    ]


@pytest.fixture
def answer_transform1() -> Callable:
    def func(inputs: np.ndarray):
        inputs = make_torch_input(inputs)
        inputs = resize(inputs, 256)
        inputs = center_crop(inputs, [224, 224])
        inputs = inputs.squeeze(0).permute(1, 2, 0).clone().numpy()
        inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB)
        inputs /= 255
        inputs = make_torch_input(inputs)
        inputs = normalize(inputs, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        inputs = inputs.squeeze(0).permute(1, 2, 0).clone().numpy()
        inputs = np.transpose(inputs, [2, 0, 1])
        return inputs

    return func


@pytest.fixture
def answer_transform2() -> Callable:
    def func(inputs: np.ndarray):
        inputs = make_torch_input(inputs)
        inputs = resize(inputs, 256)
        inputs = center_crop(inputs, [224, 224])
        inputs = inputs.squeeze(0).permute(1, 2, 0).clone().numpy()
        inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB)
        inputs = make_torch_input(inputs)
        inputs = inputs.squeeze(0).permute(1, 2, 0).clone().numpy()
        return inputs

    return func
