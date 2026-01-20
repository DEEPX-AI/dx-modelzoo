from torchvision.transforms import Compose

from dx_modelzoo.enums import DatasetType, EvaluationType
from dx_modelzoo.models import ModelBase, ModelInfo
from dx_modelzoo.models.image_classification import topk_postprocessing
from dx_modelzoo.preprocessing.centercrop import CenterCrop
from dx_modelzoo.preprocessing.convertcolor import ConvertColor
from dx_modelzoo.preprocessing.div import Div
from dx_modelzoo.preprocessing.normalize import Normalize
from dx_modelzoo.preprocessing.resize import Resize
from dx_modelzoo.preprocessing.transpose import Transpose


class MaxViT_Tiny(ModelBase):
    info = ModelInfo(
        name="MaxViT_Tiny",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=224, interpolation="BICUBIC"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=224, interpolation="BICUBIC"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                Transpose([2, 0, 1]),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing
