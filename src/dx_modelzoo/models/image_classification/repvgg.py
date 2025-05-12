from dx_modelzoo.enums import DatasetType, EvaluationType
from dx_modelzoo.models import ModelBase, ModelInfo
from dx_modelzoo.models.image_classification import topk_postprocessing


class RepVGGA1(ModelBase):
    info = ModelInfo(
        name="RepVGGA1",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
        raw_performance="74.09 91.71",
        q_lite_performance="63.08 84.94",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return [
            {
                "resize": {
                    "mode": "torchvision",
                    "size": 320,
                    "interpolation": "BILINEAR",
                }
            },
            {"centercrop": {"width": 320, "height": 320}},
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
