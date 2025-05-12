import numpy as np
import pytest
import torch

from dx_modelzoo.enums import SessionType
from dx_modelzoo.preprocessing import PreProcessingCompose


@pytest.mark.parametrize(
    "preprocessings, answer_transform, session_type",
    [
        ("preprocessings1", "answer_transform1", SessionType.onnxruntime),
        ("preprocessings1", "answer_transform2", SessionType.dxruntime),
    ],
)
def test_preprocessing_compose(preprocessings, answer_transform, session_type, request):
    preprocessings = request.getfixturevalue(preprocessings)
    answer_transform = request.getfixturevalue(answer_transform)

    compose = PreProcessingCompose(preprocessings, session_type)

    dummy_input = torch.randn(640, 640, 3)
    answer_input = dummy_input.clone()

    output = compose(dummy_input.clone().numpy())
    answer = answer_transform(answer_input.clone().numpy())

    assert output.shape == answer.shape
    assert np.allclose(output, answer)
