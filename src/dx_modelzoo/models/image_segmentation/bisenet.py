from torchvision.transforms import Compose

from dx_modelzoo.enums import DatasetType, EvaluationType
from dx_modelzoo.models import ModelBase, ModelInfo
from dx_modelzoo.preprocessing.convertcolor import ConvertColor
from dx_modelzoo.preprocessing.div import Div
from dx_modelzoo.preprocessing.normalize import Normalize
from dx_modelzoo.preprocessing.resize import Resize
from dx_modelzoo.preprocessing.transpose import Transpose


def bisenet_postprocessing(inputs):
    return inputs[0]


class BiSeNetV1(ModelBase):
    info = ModelInfo(
        name="BiSeNetV1",
        dataset=DatasetType.city,
        evaluation=EvaluationType.segmentation,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                ConvertColor("BGR2RGB"),
                Resize(width=2048, height=1024),
                Transpose([2, 0, 1]),
                Div(255),
                Normalize([0.3257, 0.369, 0.3223], [0.2112, 0.2148, 0.2115]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                ConvertColor("BGR2RGB"),
                Resize(width=2048, height=1024),
            ]
        )

    def postprocessing(self):
        return bisenet_postprocessing


class BiSeNetV2(ModelBase):
    info = ModelInfo(
        name="BiSeNetV2",
        dataset=DatasetType.city,
        evaluation=EvaluationType.segmentation,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                ConvertColor("BGR2RGB"),
                Resize(width=2048, height=1024),
                Transpose([2, 0, 1]),
                Div(255),
                Normalize([0.3257, 0.369, 0.3223], [0.2112, 0.2148, 0.2115]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                ConvertColor("BGR2RGB"),
                Resize(width=2048, height=1024),
            ]
        )

    def postprocessing(self):
        return bisenet_postprocessing
