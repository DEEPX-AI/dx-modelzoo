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


class VGG11(ModelBase):
    info = ModelInfo(
        name="VGG11",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
        raw_performance="69.03 88.63",
        q_lite_performance="68.94 88.61",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=256, interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=256, interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class VGG11BN(ModelBase):
    info = ModelInfo(
        name="VGG11BN",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
        raw_performance="70.37 89.81",
        q_lite_performance="70.24, 89.73",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=256, interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=256, interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class VGG13(ModelBase):
    info = ModelInfo(
        name="VGG13",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
        raw_performance="69.93 89.26",
        q_lite_performance="69.83 89.26",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=256, interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=256, interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class VGG13BN(ModelBase):
    info = ModelInfo(
        name="VGG13BN",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
        raw_performance="71.56 90.38",
        q_lite_performance="71.59 90.37",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=256, interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=256, interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class VGG19BN(ModelBase):
    info = ModelInfo(
        name="VGG19BN",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
        raw_performance="74.24 91.84",
        q_lite_performance="74.09 91.80",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=256, interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=256, interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing
