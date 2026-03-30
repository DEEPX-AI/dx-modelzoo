from typing import Callable, List

import numpy as np
from torchvision.transforms import Compose

from dx_modelzoo.enums import DatasetType, EvaluationType
from dx_modelzoo.models import ModelBase, ModelInfo
from dx_modelzoo.preprocessing.convertcolor import ConvertColor
from dx_modelzoo.preprocessing.div import Div
from dx_modelzoo.preprocessing.resize import Resize
from dx_modelzoo.preprocessing.transpose import Transpose


def _make_tddfa_v2_postprocessing(data_dir: str) -> Callable:
    """Create postprocessing function with lazy-loaded BFM decoder."""
    decoder = None

    def postprocessing(outputs: List[np.ndarray]):
        """Postprocessing for 3DDFA_V2 models.

        Model outputs:
            outputs[0]: [1, 62] - 3DMM parameters

        Returns:
            landmarks: [68, 2] array of 2D face landmark coordinates
        """
        nonlocal decoder
        if decoder is None:
            import os

            from dx_modelzoo.models.face_landmark.bfm_decoder import BFMDecoder

            bfm_dir = os.path.join(data_dir, "Code")
            decoder = BFMDecoder(bfm_dir)

        params = outputs[0]  # [1, 62]
        landmarks = decoder(params)  # [68, 2]
        return landmarks

    return postprocessing


class TDDFA_V2_MobileNet05(ModelBase):
    """3DDFA_V2 with MobileNet 0.5x backbone for 3D face alignment.

    Input: [1, 3, 120, 120] face crop
    Output: 62 3DMM parameters -> 68 2D face landmarks
    """

    info = ModelInfo(
        name="TDDFA_V2_MobileNet05",
        dataset=DatasetType.aflw2000_3d,
        evaluation=EvaluationType.face_landmark,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="default", size=(120, 120)),
                ConvertColor("BGR2RGB"),
                Div(255),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="default", size=(120, 120)),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return _make_tddfa_v2_postprocessing(self.evaluator.dataset.data_dir)


class TDDFA_V2_MobileNetV1(ModelBase):
    """3DDFA_V2 with MobileNetV1 backbone for 3D face alignment.

    Input: [1, 3, 120, 120] face crop
    Output: 62 3DMM parameters -> 68 2D face landmarks
    """

    info = ModelInfo(
        name="TDDFA_V2_MobileNetV1",
        dataset=DatasetType.aflw2000_3d,
        evaluation=EvaluationType.face_landmark,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="default", size=(120, 120)),
                ConvertColor("BGR2RGB"),
                Div(255),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="default", size=(120, 120)),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return _make_tddfa_v2_postprocessing(self.evaluator.dataset.data_dir)


class TDDFA_V2_ResNet22(ModelBase):
    """3DDFA_V2 with ResNet22 backbone for 3D face alignment.

    Input: [1, 3, 120, 120] face crop
    Output: 62 3DMM parameters -> 68 2D face landmarks
    """

    info = ModelInfo(
        name="TDDFA_V2_ResNet22",
        dataset=DatasetType.aflw2000_3d,
        evaluation=EvaluationType.face_landmark,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="default", size=(120, 120)),
                ConvertColor("BGR2RGB"),
                Div(255),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="default", size=(120, 120)),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return _make_tddfa_v2_postprocessing(self.evaluator.dataset.data_dir)
