from torchvision.transforms import Compose

from dx_modelzoo.enums import DatasetType, EvaluationType
from dx_modelzoo.models import ModelBase, ModelInfo
from dx_modelzoo.preprocessing.div import Div
from dx_modelzoo.preprocessing.expanddim import ExpandDim
from dx_modelzoo.preprocessing.resize import Resize
from dx_modelzoo.preprocessing.transpose import Transpose


def scdepthv3_postprocessing(inputs):
    return inputs[0]


class SCDepthV3(ModelBase):
    info = ModelInfo(name="SCDepthV3", dataset=DatasetType.nyu, evaluation=EvaluationType.depth_estimation)

    def __init__(self, evaluator):
        super().__init__(evaluator)

        self.evaluator.dataset.depth_preprocessing = self.depth_preprocessing()
        self.evaluator.use_median_scaling = True

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="default", size=(256, 320)),
                Div(255),
                Transpose([2, 0, 1]),
            ]
        )

    def depth_preprocessing(self):
        return Compose(
            [
                Resize(mode="default", size=(256, 320)),
                ExpandDim(0),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="default", size=(256, 320)),
            ]
        )

    def postprocessing(self):
        return scdepthv3_postprocessing
