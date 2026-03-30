from typing import List

import numpy as np
import torch
from torchvision.transforms import Compose

from dx_modelzoo.enums import DatasetType, EvaluationType
from dx_modelzoo.models import ModelBase, ModelInfo
from dx_modelzoo.preprocessing.convertcolor import ConvertColor
from dx_modelzoo.preprocessing.div import Div
from dx_modelzoo.preprocessing.normalize import Normalize
from dx_modelzoo.preprocessing.resize import Resize
from dx_modelzoo.preprocessing.transpose import Transpose


def face_attr_postprocessing(outputs: List[np.ndarray]):
    """Postprocessing for face attribute model.

    Model outputs:
        outputs[0]: [1, 40, 2] - logits for 40 binary attributes

    Returns:
        Tuple of (logits,)
        - logits: [40, 2] tensor
    """
    logits = outputs[0]
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)
    logits = logits.squeeze(0)  # [40, 2]
    return (logits,)


class FaceAttrResnetV1_18(ModelBase):
    """ResNet-v1-18 based face attribute recognition model.

    Input: [1, 3, 218, 178] aligned face image
    Output: 40 binary attribute predictions (logits)
    """

    info = ModelInfo(
        name="FaceAttrResnetV1_18",
        dataset=DatasetType.celeba,
        evaluation=EvaluationType.face_attribute,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="default", size=(218, 178)),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="default", size=(218, 178)),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return face_attr_postprocessing
