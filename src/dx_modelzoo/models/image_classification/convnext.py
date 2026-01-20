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
from dx_modelzoo.preprocessing.expanddim import ExpandDim


class ConvNeXtBase224_22k_1k(ModelBase):
    info = ModelInfo(
        name="ConvNeXtBase224_22k_1k",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
        raw_performance="",
        q_lite_performance="",
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


class ConvNeXtBase384_22k_1k(ModelBase):
    info = ModelInfo(
        name="ConvNeXtBase384_22k_1k",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
        raw_performance="",
        q_lite_performance="",
    )

    def __init__(self, evaluator: ICEvaluator) -> None:
        super().__init__(evaluator)

    def preprocessing(self) -> Compose:
        return Compose(
            [
                Resize(mode="torchvision", size=[384, 384], interpolation="BILINEAR"),
                CenterCrop(384, 384),
                ConvertColor("BGR2RGB"),
                Div(x=255),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                Transpose(axis=[2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=[384, 384], interpolation="BILINEAR"),
                CenterCrop(384, 384),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self) -> Callable:
        return topk_postprocessing


class ConvNeXtBase224_1k(ModelBase):
    info = ModelInfo(
        name="ConvNeXtBase224_1k",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
        raw_performance="",
        q_lite_performance="",
    )

    def __init__(self, evaluator: ICEvaluator) -> None:
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

    def postprocessing(self) -> Callable:
        return topk_postprocessing


class ConvNeXtLarge224_1k(ModelBase):
    info = ModelInfo(
        name="ConvNeXtLarge224_1k",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
        raw_performance="",
        q_lite_performance="",
    )

    def __init__(self, evaluator: ICEvaluator) -> None:
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

    def postprocessing(self) -> Callable:
        return topk_postprocessing


class ConvNeXtSmall224_1k(ModelBase):
    info = ModelInfo(
        name="ConvNeXtSmall224_1k",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
        raw_performance="",
        q_lite_performance="",
    )

    def __init__(self, evaluator: ICEvaluator) -> None:
        super().__init__(evaluator)

    def preprocessing(self) -> Compose:
        return Compose(
            [
                Resize(mode="torchvision", size=[230, 230], interpolation="BILINEAR"),
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
                Resize(mode="torchvision", size=[230, 230], interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self) -> Callable:
        return topk_postprocessing



class ConvNeXtTiny224_1k(ModelBase):
    info = ModelInfo(
        name="ConvNeXtTiny224_1k",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
        raw_performance="",
        q_lite_performance="",
    )

    def __init__(self, evaluator: ICEvaluator) -> None:
        super().__init__(evaluator)

    def preprocessing(self) -> Compose:
        return Compose(
            [
                Resize(mode="torchvision", size=[236, 236], interpolation="BILINEAR"),
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
                Resize(mode="torchvision", size=[236, 236], interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self) -> Callable:
        return topk_postprocessing



