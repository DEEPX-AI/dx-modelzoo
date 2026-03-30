import numpy as np
from torchvision.transforms import Compose

from dx_modelzoo.enums import DatasetType, EvaluationType
from dx_modelzoo.models import ModelBase, ModelInfo
from dx_modelzoo.preprocessing.convertcolor import ConvertColor
from dx_modelzoo.preprocessing.div import Div
from dx_modelzoo.preprocessing.resize import Resize
from dx_modelzoo.preprocessing.transpose import Transpose


def _zero_dce_postprocessing(outputs):
    """Post-processing for Zero-DCE: output is already the enhanced image [1,3,H,W] in [0,1]."""
    out = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
    return np.clip(out.astype(np.float32), 0.0, 1.0)


class ZeroDCE(ModelBase):
    """Zero-DCE for low-light image enhancement.

    Input: [1, 3, 400, 600] low-light image (RGB, float32, [0,1])
    Output: [1, 3, 400, 600] enhanced image (RGB, float32, [0,1])
    """

    info = ModelInfo(
        name="ZeroDCE",
        dataset=DatasetType.lol,
        evaluation=EvaluationType.lol,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="default", size=(400, 600)),
                ConvertColor("BGR2RGB"),
                Div(255),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="default", size=(400, 600)),
                ConvertColor("BGR2RGB"),
                Div(255),
                Transpose([2, 0, 1]),
            ]
        )

    def postprocessing(self):
        return _zero_dce_postprocessing
