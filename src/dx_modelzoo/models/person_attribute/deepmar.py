from typing import List

import numpy as np
from torchvision.transforms import Compose

from dx_modelzoo.enums import DatasetType, EvaluationType
from dx_modelzoo.models import ModelBase, ModelInfo
from dx_modelzoo.preprocessing.convertcolor import ConvertColor
from dx_modelzoo.preprocessing.div import Div
from dx_modelzoo.preprocessing.normalize import Normalize
from dx_modelzoo.preprocessing.resize import Resize
from dx_modelzoo.preprocessing.transpose import Transpose


def deepmar_postprocessing(outputs: List[np.ndarray]):
    return outputs[0]


class DeepMAR_ResNet18(ModelBase):
    """DeepMAR ResNet-v1-18 for pedestrian attribute recognition.

    Input: [1, 3, 224, 224]
    Output: [1, 35] sigmoid logits for 35 PETA attributes
    """

    info = ModelInfo(
        name="DeepMAR_ResNet18",
        dataset=DatasetType.peta,
        evaluation=EvaluationType.person_attribute,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(width=224, height=224),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(width=224, height=224),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return deepmar_postprocessing


class DeepMAR_ResNet50(ModelBase):
    """DeepMAR ResNet-50 for pedestrian attribute recognition.

    Input: [1, 3, 224, 224]
    Output: [1, 35] sigmoid logits for 35 PETA attributes
    """

    info = ModelInfo(
        name="DeepMAR_ResNet50",
        dataset=DatasetType.peta,
        evaluation=EvaluationType.person_attribute,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(width=224, height=224),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(width=224, height=224),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return deepmar_postprocessing
