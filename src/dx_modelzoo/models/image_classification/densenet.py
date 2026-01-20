from torchvision.transforms import Compose

from dx_modelzoo.enums import DatasetType, EvaluationType
from dx_modelzoo.models import ModelBase, ModelInfo
from dx_modelzoo.models.image_classification import topk_postprocessing
from dx_modelzoo.preprocessing.centercrop import CenterCrop
from dx_modelzoo.preprocessing.convertcolor import ConvertColor
from dx_modelzoo.preprocessing.div import Div
from dx_modelzoo.preprocessing.normalize import Normalize
from dx_modelzoo.preprocessing.resize import Resize
from dx_modelzoo.preprocessing.transpose import Transpose


class DenseNet121(ModelBase):
    info = ModelInfo(
        name="DenseNet121",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
        raw_performance="74.43 91.97",
        q_lite_performance="72.68 90.87",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize("torchvision", size=256, interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize("torchvision", size=256, interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class DenseNet161(ModelBase):
    info = ModelInfo(
        name="DenseNet161",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
        raw_performance="77.11 93.56",
        q_lite_performance="76.41 93.07",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize("torchvision", size=256, interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize("torchvision", size=256, interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing

class DenseNet169(ModelBase):
    info = ModelInfo(
        name="DenseNet169",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
        raw_performance="",
        q_lite_performance="",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize("torchvision", size=256, interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize("torchvision", size=256, interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing

class DenseNet201(ModelBase):
    info = ModelInfo(
        name="DenseNet201",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
        raw_performance="",
        q_lite_performance="",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize("torchvision", size=256, interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize("torchvision", size=256, interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing
