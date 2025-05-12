from unittest import mock

import numpy as np
import onnxruntime as ort
import pytest
import torch

from dx_modelzoo.session.onnx_runtime_session import OnnxRuntimeSession, get_ort_provider, get_ort_session_options


@pytest.mark.parametrize("return_value", [True, False])
def test_get_ort_provider(return_value):
    with mock.patch("torch.cuda.is_available") as cuda_mock:
        cuda_mock.return_value = return_value
        provider = get_ort_provider()

        if return_value:
            assert provider[0] == "CUDAExecutionProvider"
        else:
            assert provider[0] == "CPUExecutionProvider"


def test_get_ort_session_options():
    session_options = get_ort_session_options()
    assert isinstance(session_options, ort.SessionOptions)
    assert session_options.graph_optimization_level == ort.GraphOptimizationLevel.ORT_ENABLE_BASIC


def test_onnx_runtime(test_onnx_path):
    dummy_input = torch.randn(1, 3, 10, 10)

    session = OnnxRuntimeSession(test_onnx_path)

    output = session.run(dummy_input)
    answer = (dummy_input + 1) * 2

    assert isinstance(output, list)
    assert np.allclose(output, answer)
