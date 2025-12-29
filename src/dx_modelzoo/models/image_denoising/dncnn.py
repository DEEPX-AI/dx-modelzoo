from typing import List

import numpy as np
from torchvision.transforms import Compose

from dx_modelzoo.enums import DatasetType, EvaluationType
from dx_modelzoo.models import ModelBase, ModelInfo
from dx_modelzoo.preprocessing.convertcolor import ConvertColor
from dx_modelzoo.preprocessing.div import Div
from dx_modelzoo.preprocessing.expanddim import ExpandDim


def dcnn_postprocessing(outputs: List[np.array]):
    return outputs[0]


class DnCNN_15(ModelBase):
    info = ModelInfo(name="DnCNN_15", dataset=DatasetType.bsd68, evaluation=EvaluationType.bsd68)

    def __init__(self, evaluator):
        super().__init__(evaluator)
        self.evaluator.noise_level = 15
        self.evaluator.input_size = (512, 512)

    def preprocessing(self):
        return Compose([ConvertColor("BGR2GRAY"), Div(255), ExpandDim(0)])

    def npu_preprocessing(self):
        return Compose([ConvertColor("BGR2GRAY"), ExpandDim(0)])

    def postprocessing(self):
        return dcnn_postprocessing


class DnCNN_25(ModelBase):
    info = ModelInfo(name="DnCNN_15", dataset=DatasetType.bsd68, evaluation=EvaluationType.bsd68)

    def __init__(self, evaluator):
        super().__init__(evaluator)
        self.evaluator.noise_level = 25
        self.evaluator.input_size = (512, 512)

    def preprocessing(self):
        return Compose([ConvertColor("BGR2GRAY"), Div(255), ExpandDim(0)])

    def npu_preprocessing(self):
        return Compose([ConvertColor("BGR2GRAY"), ExpandDim(0)])

    def postprocessing(self):
        return dcnn_postprocessing


class DnCNN_50(ModelBase):
    info = ModelInfo(name="DnCNN_15", dataset=DatasetType.bsd68, evaluation=EvaluationType.bsd68)

    def __init__(self, evaluator):
        super().__init__(evaluator)
        self.evaluator.noise_level = 50
        self.evaluator.input_size = (512, 512)

    def preprocessing(self):
        return Compose([ConvertColor("BGR2GRAY"), Div(255), ExpandDim(0)])

    def npu_preprocessing(self):
        return Compose([ConvertColor("BGR2GRAY"), ExpandDim(0)])

    def postprocessing(self):
        return dcnn_postprocessing
