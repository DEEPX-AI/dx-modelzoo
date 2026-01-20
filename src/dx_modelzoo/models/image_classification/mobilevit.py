from torchvision.transforms import Compose

from dx_modelzoo.enums import DatasetType, EvaluationType
from dx_modelzoo.models import ModelBase, ModelInfo
from dx_modelzoo.models.image_classification import topk_postprocessing
from dx_modelzoo.preprocessing.centercrop import CenterCrop
from dx_modelzoo.preprocessing.convertcolor import ConvertColor
from dx_modelzoo.preprocessing.div import Div
from dx_modelzoo.preprocessing.resize import Resize
from dx_modelzoo.preprocessing.transpose import Transpose


class MobileViT_Small(ModelBase):
    info = ModelInfo(
        name="MobileViT_Small",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=288, interpolation="BICUBIC"),
                CenterCrop(256, 256),
                ConvertColor("BGR2RGB"),
                Div(255),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=288, interpolation="BICUBIC"),
                CenterCrop(256, 256),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class MobileViT_X_Small(ModelBase):
    info = ModelInfo(
        name="MobileViT_X_Small",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=288, interpolation="BICUBIC"),
                CenterCrop(256, 256),
                ConvertColor("BGR2RGB"),
                Div(255),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=288, interpolation="BICUBIC"),
                CenterCrop(256, 256),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class MobileViT_XX_Small(ModelBase):
    info = ModelInfo(
        name="MobileViT_XX_Small",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=288, interpolation="BICUBIC"),
                CenterCrop(256, 256),
                ConvertColor("BGR2RGB"),
                Div(255),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=288, interpolation="BICUBIC"),
                CenterCrop(256, 256),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing
