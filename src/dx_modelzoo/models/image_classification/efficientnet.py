from dx_modelzoo.enums import DatasetType, EvaluationType
from dx_modelzoo.models import ModelBase, ModelInfo
from dx_modelzoo.models.image_classification import topk_postprocessing


class EfficientNetB2(ModelBase):
    info = ModelInfo(
        name="EfficientNetB2",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
        raw_performance="80.61 95.31",
        q_lite_performance="79.19 94.55",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return [
            {
                "resize": {
                    "mode": "torchvision",
                    "size": 288,
                    "interpolation": "BICUBIC",
                }
            },
            {"centercrop": {"width": 288, "height": 288}},
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


class EfficientNetV2S(ModelBase):
    info = ModelInfo(
        name="EfficientNetV2S",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
        raw_performance="84.24 96.87",
        q_lite_performance="80.5 95.20",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return [
            {
                "resize": {
                    "mode": "torchvision",
                    "size": 384,
                    "interpolation": "BILINEAR",
                }
            },
            {"centercrop": {"width": 384, "height": 384}},
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
