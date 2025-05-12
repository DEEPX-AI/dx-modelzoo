from dx_modelzoo.enums import DatasetType, EvaluationType
from dx_modelzoo.models import ModelBase, ModelInfo
from dx_modelzoo.models.image_classification import topk_postprocessing


class OSNet0_5(ModelBase):
    info = ModelInfo(
        name="OSNet0_5",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
        raw_performance="69.45 89.13",
        q_lite_performance="67.00 87.65",
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


class OSNet0_25(ModelBase):
    info = ModelInfo(
        name="OSNet0_25",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
        raw_performance="58.34 81.20",
        q_lite_performance="53.64 78.03",
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
