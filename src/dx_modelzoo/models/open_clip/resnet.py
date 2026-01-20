from torchvision.transforms import Compose

from dx_modelzoo.enums import DatasetType, EvaluationType
from dx_modelzoo.models import ModelBase, ModelInfo
from dx_modelzoo.preprocessing.centercrop import CenterCrop
from dx_modelzoo.preprocessing.convertcolor import ConvertColor
from dx_modelzoo.preprocessing.div import Div
from dx_modelzoo.preprocessing.normalize import Normalize
from dx_modelzoo.preprocessing.resize import Resize
from dx_modelzoo.preprocessing.transpose import Transpose


class RN50x16_openai(ModelBase):
    info = ModelInfo(
        name="RN50x16_openai",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.zeroshot_classification,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(width=384, height=384),
                CenterCrop(height=384, width=384),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(width=384, height=384),
                CenterCrop(height=384, width=384),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        def open_clip_postprocessing(output):
            return output[0]

        return open_clip_postprocessing


class RN50x64_openai(ModelBase):
    info = ModelInfo(
        name="RN50x64_openai",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.zeroshot_classification,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(width=448, height=448),
                CenterCrop(height=448, width=448),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(width=448, height=448),
                CenterCrop(height=448, width=448),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        def open_clip_postprocessing(output):
            return output[0]

        return open_clip_postprocessing
