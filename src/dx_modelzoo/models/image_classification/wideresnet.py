from dx_modelzoo.enums import DatasetType, EvaluationType
from dx_modelzoo.models import ModelBase, ModelInfo
from dx_modelzoo.models.image_classification import topk_postprocessing


class WideResNet101_2(ModelBase):
    info = ModelInfo(
        name="WideResNet101",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
        raw_performance="82.52 96.01",
        q_lite_performance="82.30 95.99",
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


class WideResNet50_2(ModelBase):
    info = ModelInfo(
        name="WideResNet50",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
        raw_performance="81.61 95.76",
        q_lite_performance="81.54 95.74",
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
