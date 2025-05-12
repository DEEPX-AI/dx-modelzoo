from typing import Callable, Dict, List

from dx_modelzoo.enums import DatasetType, EvaluationType
from dx_modelzoo.evaluator.ic_evaluator import ICEvaluator
from dx_modelzoo.models import ModelBase, ModelInfo
from dx_modelzoo.models.image_classification import topk_postprocessing

PREPROCESSING = Dict[str, Dict[str, str | int | float | List[int | float]]]


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

    def preprocessing(self) -> List[PREPROCESSING]:
        return [
            {
                "resize": {
                    "mode": "torchvision",
                    "size": 256,
                    "interpolation": "BILINEAR",
                }
            },
            {"centercrop": {"width": 224, "height": 224}},
            {"convertColor": {"form": "BGR2RGB"}},
            {"div": {"x": 255}},
            {
                "normalize": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                }
            },
            {"transpose": {"axis": [2, 0, 1]}},
            {"expandDim": {"axis": 0}},
        ]

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

    def preprocessing(self):
        return [
            {
                "resize": {
                    "mode": "torchvision",
                    "size": 256,
                    "interpolation": "BILINEAR",
                }
            },
            {"centercrop": {"width": 224, "height": 224}},
            {"convertColor": {"form": "BGR2RGB"}},
            {"div": {"x": 255}},
            {
                "normalize": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                }
            },
            {"transpose": {"axis": [2, 0, 1]}},
            {"expandDim": {"axis": 0}},
        ]

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

    def preprocessing(self):
        return [
            {
                "resize": {
                    "mode": "torchvision",
                    "size": 232,
                    "interpolation": "BILINEAR",
                }
            },
            {"centercrop": {"width": 224, "height": 224}},
            {"convertColor": {"form": "BGR2RGB"}},
            {"div": {"x": 255}},
            {
                "normalize": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                }
            },
            {"transpose": {"axis": [2, 0, 1]}},
            {"expandDim": {"axis": 0}},
        ]

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

    def preprocessing(self):
        return [
            {
                "resize": {
                    "mode": "torchvision",
                    "size": 232,
                    "interpolation": "BILINEAR",
                }
            },
            {"centercrop": {"width": 224, "height": 224}},
            {"convertColor": {"form": "BGR2RGB"}},
            {"div": {"x": 255}},
            {
                "normalize": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                }
            },
            {"transpose": {"axis": [2, 0, 1]}},
            {"expandDim": {"axis": 0}},
        ]

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

    def preprocessing(self):
        return [
            {
                "resize": {
                    "mode": "torchvision",
                    "size": 232,
                    "interpolation": "BILINEAR",
                }
            },
            {"centercrop": {"width": 224, "height": 224}},
            {"convertColor": {"form": "BGR2RGB"}},
            {"div": {"x": 255}},
            {
                "normalize": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                }
            },
            {"transpose": {"axis": [2, 0, 1]}},
            {"expandDim": {"axis": 0}},
        ]

    def postprocessing(self):
        return topk_postprocessing
