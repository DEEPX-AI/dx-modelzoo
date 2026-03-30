from typing import Callable

from torchvision.transforms import Compose

from dx_modelzoo.enums import DatasetType, EvaluationType
from dx_modelzoo.evaluator.ic_evaluator import ICEvaluator
from dx_modelzoo.models import ModelBase, ModelInfo
from dx_modelzoo.models.image_classification import topk_postprocessing
from dx_modelzoo.preprocessing.centercrop import CenterCrop
from dx_modelzoo.preprocessing.convertcolor import ConvertColor
from dx_modelzoo.preprocessing.div import Div
from dx_modelzoo.preprocessing.resize import Resize
from dx_modelzoo.preprocessing.transpose import Transpose


class YOLO26n_cls(ModelBase):
    info = ModelInfo(name="YOLO26n_cls", dataset=DatasetType.imagenet, evaluation=EvaluationType.image_classification)

    def __init__(self, evaluator: ICEvaluator):
        super().__init__(evaluator)

    def preprocessing(self) -> Compose:
        return Compose(
            [
                Resize(mode="torchvision", size=224, interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
                Div(x=255),
                Transpose(axis=[2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=224, interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self) -> Callable:
        return topk_postprocessing


class YOLO26s_cls(ModelBase):
    info = ModelInfo(name="YOLO26n_cls", dataset=DatasetType.imagenet, evaluation=EvaluationType.image_classification)

    def __init__(self, evaluator: ICEvaluator):
        super().__init__(evaluator)

    def preprocessing(self) -> Compose:
        return Compose(
            [
                Resize(mode="torchvision", size=224, interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
                Div(x=255),
                Transpose(axis=[2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=224, interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self) -> Callable:
        return topk_postprocessing


class YOLO26m_cls(ModelBase):
    info = ModelInfo(name="YOLO26n_cls", dataset=DatasetType.imagenet, evaluation=EvaluationType.image_classification)

    def __init__(self, evaluator: ICEvaluator):
        super().__init__(evaluator)

    def preprocessing(self) -> Compose:
        return Compose(
            [
                Resize(mode="torchvision", size=224, interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
                Div(x=255),
                Transpose(axis=[2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=224, interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self) -> Callable:
        return topk_postprocessing


class YOLO26l_cls(ModelBase):
    info = ModelInfo(name="YOLO26n_cls", dataset=DatasetType.imagenet, evaluation=EvaluationType.image_classification)

    def __init__(self, evaluator: ICEvaluator):
        super().__init__(evaluator)

    def preprocessing(self) -> Compose:
        return Compose(
            [
                Resize(mode="torchvision", size=224, interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
                Div(x=255),
                Transpose(axis=[2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=224, interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self) -> Callable:
        return topk_postprocessing


class YOLO26x_cls(ModelBase):
    info = ModelInfo(name="YOLO26n_cls", dataset=DatasetType.imagenet, evaluation=EvaluationType.image_classification)

    def __init__(self, evaluator: ICEvaluator):
        super().__init__(evaluator)

    def preprocessing(self) -> Compose:
        return Compose(
            [
                Resize(mode="torchvision", size=224, interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
                Div(x=255),
                Transpose(axis=[2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=224, interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self) -> Callable:
        return topk_postprocessing
