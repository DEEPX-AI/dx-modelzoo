import cv2
import numpy as np
import pytest
import torch

from dx_modelzoo.preprocessing.convertcolor import FORM_TO_CODE_DICT, ConvertColor


@pytest.mark.parametrize("form", ["BGR2RGB"])
def test_convertcolor(form):
    convertcolor = ConvertColor(form)

    dummy_input = torch.randn(224, 224, 3).numpy()

    output = convertcolor(dummy_input)
    answer = cv2.cvtColor(dummy_input, FORM_TO_CODE_DICT[form])

    assert output.shape == answer.shape
    assert np.allclose(output, answer)
