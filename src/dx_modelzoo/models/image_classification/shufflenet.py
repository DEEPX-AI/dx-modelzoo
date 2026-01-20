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


class ShuffleNetV1_x1_0(ModelBase):
    info = ModelInfo(
        name="ShuffleNetV1_x1_0",
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
                Resize(mode="torchvision", size=256, interpolation="BILINEAR"),
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
                Resize(mode="torchvision", size=256, interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self) -> Callable:
        return topk_postprocessing


class ShuffleNetV2_x0_5(ModelBase):
    info = ModelInfo(
        name="ShuffleNetV2_x0_5",
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
                Resize(mode="torchvision", size=256, interpolation="BILINEAR"),
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
                Resize(mode="torchvision", size=256, interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self) -> Callable:
        return topk_postprocessing


class ShuffleNetV2_x1_0(ModelBase):
    info = ModelInfo(
        name="ShuffleNetV2_x1_0",
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
                Resize(mode="torchvision", size=256, interpolation="BILINEAR"),
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
                Resize(mode="torchvision", size=256, interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self) -> Callable:
        return topk_postprocessing


class ShuffleNetV2_x1_5(ModelBase):
    info = ModelInfo(
        name="ShuffleNetV2_x1_5",
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
                Resize(mode="torchvision", size=232, interpolation="BILINEAR"),
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
                Resize(mode="torchvision", size=232, interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self) -> Callable:
        return topk_postprocessing


class ShuffleNetV2_x2_0(ModelBase):
    info = ModelInfo(
        name="ShuffleNetV2_x2_0",
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
                Resize(mode="torchvision", size=232, interpolation="BILINEAR"),
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
                Resize(mode="torchvision", size=232, interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self) -> Callable:
        return topk_postprocessing
