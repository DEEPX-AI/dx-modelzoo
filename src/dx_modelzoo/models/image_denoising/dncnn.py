from typing import List

import numpy as np

from dx_modelzoo.enums import DatasetType, EvaluationType
from dx_modelzoo.models import ModelBase, ModelInfo


def dcnn_postprocessing(outputs: List[np.array]):
    return outputs[0]


class DnCNN_15(ModelBase):
    info = ModelInfo(name="DnCNN_15", dataset=DatasetType.bsd68, evaluation=EvaluationType.bsd68)

    def __init__(self, evaluator):
        super().__init__(evaluator)
        self.evaluator.noise_level = 15
        self.evaluator.input_size = (512, 512)

    def preprocessing(self):
        return [
            {"convertColor": {"form": "BGR2GRAY"}},
            {"div": {"x": 255.0}},
            {"expandDim": {"axis": 0}},
            {"expandDim": {"axis": 0}},
        ]

    def postprocessing(self):
        return dcnn_postprocessing


class DnCNN_25(ModelBase):
    info = ModelInfo(name="DnCNN_15", dataset=DatasetType.bsd68, evaluation=EvaluationType.bsd68)

    def __init__(self, evaluator):
        super().__init__(evaluator)
        self.evaluator.noise_level = 25
        self.evaluator.input_size = (512, 512)

    def preprocessing(self):
        return [
            {"convertColor": {"form": "BGR2GRAY"}},
            {"div": {"x": 255.0}},
            {"expandDim": {"axis": 0}},
            {"expandDim": {"axis": 0}},
        ]

    def postprocessing(self):
        return dcnn_postprocessing


class DnCNN_50(ModelBase):
    info = ModelInfo(name="DnCNN_15", dataset=DatasetType.bsd68, evaluation=EvaluationType.bsd68)

    def __init__(self, evaluator):
        super().__init__(evaluator)
        self.evaluator.noise_level = 50
        self.evaluator.input_size = (512, 512)

    def preprocessing(self):
        return [
            {"convertColor": {"form": "BGR2GRAY"}},
            {"div": {"x": 255.0}},
            {"expandDim": {"axis": 0}},
            {"expandDim": {"axis": 0}},
        ]

    def postprocessing(self):
        return dcnn_postprocessing
