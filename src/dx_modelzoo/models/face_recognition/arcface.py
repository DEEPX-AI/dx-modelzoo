from torchvision.transforms import Compose

from dx_modelzoo.enums import DatasetType, EvaluationType
from dx_modelzoo.models import ModelBase, ModelInfo
from dx_modelzoo.preprocessing.centercrop import CenterCrop
from dx_modelzoo.preprocessing.convertcolor import ConvertColor
from dx_modelzoo.preprocessing.normalize import Normalize
from dx_modelzoo.preprocessing.transpose import Transpose


def arcface_postprocessing(outputs):
    return outputs[0]


class ArcFace_MobileFaceNet(ModelBase):
    info = ModelInfo(
        name="ArcFace_MobileFaceNet",
        dataset=DatasetType.lfw,
        evaluation=EvaluationType.lfw,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                CenterCrop(112, 112),
                ConvertColor("BGR2RGB"),
                Normalize(mean=[127.5, 127.5, 127.5], std=[128.0, 128.0, 128.0]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                CenterCrop(112, 112),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return arcface_postprocessing


class ArcFace_IResNet100_MS1M(ModelBase):
    info = ModelInfo(
        name="ArcFace_IResNet100_MS1M",
        dataset=DatasetType.lfw,
        evaluation=EvaluationType.lfw,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                CenterCrop(112, 112),
                ConvertColor("BGR2RGB"),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                CenterCrop(112, 112),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return arcface_postprocessing


class ArcFace_IResNet50_MS1M(ModelBase):
    info = ModelInfo(
        name="ArcFace_IResNet50_MS1M",
        dataset=DatasetType.lfw,
        evaluation=EvaluationType.lfw,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                CenterCrop(112, 112),
                ConvertColor("BGR2RGB"),
                Normalize(mean=[127.5, 127.5, 127.5], std=[128.0, 128.0, 128.0]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                CenterCrop(112, 112),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return arcface_postprocessing


class ArcFace_R50(ModelBase):
    info = ModelInfo(
        name="ArcFace_R50",
        dataset=DatasetType.lfw,
        evaluation=EvaluationType.lfw,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                CenterCrop(112, 112),
                ConvertColor("BGR2RGB"),
                Normalize(mean=[127.5, 127.5, 127.5], std=[128.0, 128.0, 128.0]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                CenterCrop(112, 112),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return arcface_postprocessing
