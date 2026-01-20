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


class MnasNet0_5(ModelBase):
    info = ModelInfo(
        name="MnasNet0_5",
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
                Resize(mode="torchvision", size=256, interpolation="BILINEAR"),
                CenterCrop(height=224, width=224),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=256, interpolation="BILINEAR"),
                CenterCrop(height=224, width=224),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class MnasNet0_75(ModelBase):
    info = ModelInfo(
        name="MnasNet0_75",
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
                Resize(mode="torchvision", size=232, interpolation="BILINEAR"),
                CenterCrop(height=224, width=224),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=232, interpolation="BILINEAR"),
                CenterCrop(height=224, width=224),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class MnasNet1_0(ModelBase):
    info = ModelInfo(
        name="MnasNet1_0",
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
                Resize(mode="torchvision", size=256, interpolation="BILINEAR"),
                CenterCrop(height=224, width=224),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=256, interpolation="BILINEAR"),
                CenterCrop(height=224, width=224),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class MnasNet1_3(ModelBase):
    info = ModelInfo(
        name="MnasNet1_3",
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
                Resize(mode="torchvision", size=232, interpolation="BILINEAR"),
                CenterCrop(height=224, width=224),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=232, interpolation="BILINEAR"),
                CenterCrop(height=224, width=224),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing



