from dx_modelzoo.enums import DatasetType, EvaluationType
from dx_modelzoo.models import ModelBase, ModelInfo


def bisenet_postprocessing(inputs):
    return inputs[0]


class BiSeNetV1(ModelBase):
    info = ModelInfo(
        name="BiSeNetV1",
        dataset=DatasetType.city,
        evaluation=EvaluationType.segmentation,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return [
            {"convertColor": {"form": "BGR2RGB"}},
            {"resize": {"mode": "default", "width": 2048, "height": 1024}},
            {"transpose": {"axis": [2, 0, 1]}},
            {"div": {"x": 255}},
            {
                "normalize": {
                    "mean": [0.3257, 0.369, 0.3223],
                    "std": [0.2112, 0.2148, 0.2115],
                }
            },
            {"expandDim": {"axis": 0}},
        ]

    def postprocessing(self):
        return bisenet_postprocessing


class BiSeNetV2(ModelBase):
    info = ModelInfo(
        name="BiSeNetV2",
        dataset=DatasetType.city,
        evaluation=EvaluationType.segmentation,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return [
            {"convertColor": {"form": "BGR2RGB"}},
            {"resize": {"mode": "default", "width": 2048, "height": 1024}},
            {"transpose": {"axis": [2, 0, 1]}},
            {"div": {"x": 255}},
            {
                "normalize": {
                    "mean": [0.3257, 0.369, 0.3223],
                    "std": [0.2112, 0.2148, 0.2115],
                }
            },
            {"expandDim": {"axis": 0}},
        ]

    def postprocessing(self):
        return bisenet_postprocessing
