from typing import Callable

from torchvision.transforms import Compose

from dx_modelzoo.enums import DatasetType, EvaluationType
from dx_modelzoo.evaluator.ic_evaluator import ICEvaluator
from dx_modelzoo.models import ModelBase, ModelInfo
from dx_modelzoo.models.image_classification import topk_postprocessing
from dx_modelzoo.preprocessing.centercrop import CenterCrop
from dx_modelzoo.preprocessing.convertcolor import ConvertColor
from dx_modelzoo.preprocessing.div import Div
from dx_modelzoo.preprocessing.normalize import Normalize
from dx_modelzoo.preprocessing.resize import Resize
from dx_modelzoo.preprocessing.transpose import Transpose


class ResNet18(ModelBase):
    info = ModelInfo(
        name="ResNet18",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
        raw_performance="69.75 89.08",
        q_lite_performance="69.57 88.97",
    )

    def __init__(self, evaluator: ICEvaluator) -> None:
        super().__init__(evaluator)

    def preprocessing(self) -> Compose:
        return Compose(
            [
                Resize(mode="torchvision", size=[256, 256], interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
                Div(x=255),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                Transpose(axis=[2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=[256, 256], interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self) -> Callable:
        return topk_postprocessing


class ResNet34(ModelBase):
    info = ModelInfo(
        name="ResNet34",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
        raw_performance="73.29 91.41",
        q_lite_performance="73.28, 91.39",
    )

    def __init__(self, evaluator: ICEvaluator):
        super().__init__(evaluator)

    def preprocessing(self) -> Compose:
        return Compose(
            [
                Resize(mode="torchvision", size=[256, 256], interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
                Div(x=255),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                Transpose(axis=[2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=[256, 256], interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class ResNet50(ModelBase):
    info = ModelInfo(
        name="ResNet50",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
        raw_performance="80.85 95.43",
        q_lite_performance="80.65 95.33",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self) -> Compose:
        return Compose(
            [
                Resize(mode="torchvision", size=[232, 232], interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
                Div(x=255),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                Transpose(axis=[2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=[232, 232], interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class ResNet101(ModelBase):
    info = ModelInfo(
        name="ResNet101",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
        raw_performance="81.90 95.77",
        q_lite_performance="81.62 95.68",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self) -> Compose:
        return Compose(
            [
                Resize(mode="torchvision", size=[232, 232], interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
                Div(x=255),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                Transpose(axis=[2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=[232, 232], interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class ResNet152(ModelBase):
    info = ModelInfo(
        name="ResNet152",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
        raw_performance="82.28, 96.00",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self) -> Compose:
        return Compose(
            [
                Resize(mode="torchvision", size=[256, 256], interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
                Div(x=255),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                Transpose(axis=[2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=[256, 256], interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing
