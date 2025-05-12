from dx_modelzoo.enums import DatasetType, EvaluationType
from dx_modelzoo.models import ModelBase, ModelInfo
from dx_modelzoo.models.image_classification import topk_postprocessing


class MobileNetV1(ModelBase):
    info = ModelInfo(
        name="MobileNetV1",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
        raw_performance="69.49 89.55",
        q_lite_performance="68.84 89.09",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return [
            {"convertColor": {"form": "BGR2RGB"}},
            {"resize": {"width": 256, "height": 256}},
            {"centercrop": {"width": 224, "height": 224}},
            {"div": {"x": 255.0}},
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


class MobileNetV2(ModelBase):
    info = ModelInfo(
        name="MobileNetV2",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
        raw_performance="72.14 90.82",
        q_lite_performance="71.87 90.77",
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


class MobileNetV3Small(ModelBase):
    info = ModelInfo(
        name="MobileNetV3Small",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
        raw_performance="67.65 87.41",
        q_lite_performance="25.24 45.18",
    )

    def __init__(self, evaluator):
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


class MobileNetV3Large(ModelBase):
    info = ModelInfo(
        name="MobileNetV3Large",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
        raw_performance="75.26 92.57",
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
