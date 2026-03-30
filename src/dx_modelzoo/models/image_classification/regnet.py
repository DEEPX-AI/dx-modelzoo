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


class RegNetX400MF(ModelBase):
    info = ModelInfo(
        name="RegNetX400MF",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
        raw_performance="74.88 92.31",
        q_lite_performance="74.46 92.17",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=232, interpolation="BILINEAR"),
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
                Resize(mode="torchvision", size=232, interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class RegNetX800MF(ModelBase):
    info = ModelInfo(
        name="RegNetX800MF",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
        raw_performance="77.52 93.83",
        q_lite_performance="77.26 93.80",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=232, interpolation="BILINEAR"),
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
                Resize(mode="torchvision", size=232, interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class RegNetY200MF(ModelBase):
    info = ModelInfo(
        name="RegNetY200MF",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
        raw_performance="70.36 89.61",
        q_lite_performance="70.13 89.60",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="pycls", size=256, interpolation="LINEAR"),
                CenterCrop(224, 224),
                Div(255),
                Normalize([0.406, 0.456, 0.485], [0.225, 0.224, 0.229]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="pycls", size=256, interpolation="LINEAR"),
                CenterCrop(224, 224),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class RegNetY400MF(ModelBase):
    info = ModelInfo(
        name="RegNetY400MF",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
        raw_performance="75.78 92.75",
        q_lite_performance="75.38 92.71",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=232, interpolation="BILINEAR"),
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
                Resize(mode="torchvision", size=232, interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class RegNetY800MF(ModelBase):
    info = ModelInfo(
        name="RegNetY800MF",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
        raw_performance="78.83 94.49",
        q_lite_performance="78.54 94.35",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=232, interpolation="BILINEAR"),
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
                Resize(mode="torchvision", size=232, interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class RegNetX_1_6GF(ModelBase):
    info = ModelInfo(name="RegNetX_1_6GF", dataset=DatasetType.imagenet, evaluation=EvaluationType.image_classification)

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=232, interpolation="BILINEAR"),
                CenterCrop(width=224, height=224),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                Transpose(axis=[2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=232, interpolation="BILINEAR"),
                CenterCrop(width=224, height=224),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class RegNetX_1_6GF_3(ModelBase):
    info = ModelInfo(
        name="RegNetX_1_6GF_3", dataset=DatasetType.imagenet, evaluation=EvaluationType.image_classification
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=232, interpolation="BILINEAR"),
                CenterCrop(width=224, height=224),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                Transpose(axis=[2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=232, interpolation="BILINEAR"),
                CenterCrop(width=224, height=224),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class RegNetX_16GF(ModelBase):
    info = ModelInfo(name="RegNetX_16GF", dataset=DatasetType.imagenet, evaluation=EvaluationType.image_classification)

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=232, interpolation="BILINEAR"),
                CenterCrop(width=224, height=224),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                Transpose(axis=[2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=232, interpolation="BILINEAR"),
                CenterCrop(width=224, height=224),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class RegNetX_3_2GF(ModelBase):
    info = ModelInfo(name="RegNetX_3_2GF", dataset=DatasetType.imagenet, evaluation=EvaluationType.image_classification)

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=232, interpolation="BILINEAR"),
                CenterCrop(width=224, height=224),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                Transpose(axis=[2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=232, interpolation="BILINEAR"),
                CenterCrop(width=224, height=224),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class RegNetX_32GF(ModelBase):
    info = ModelInfo(name="RegNetX_32GF", dataset=DatasetType.imagenet, evaluation=EvaluationType.image_classification)

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=232, interpolation="BILINEAR"),
                CenterCrop(width=224, height=224),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                Transpose(axis=[2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=232, interpolation="BILINEAR"),
                CenterCrop(width=224, height=224),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class RegNetX_8GF(ModelBase):
    info = ModelInfo(name="RegNetX_8GF", dataset=DatasetType.imagenet, evaluation=EvaluationType.image_classification)

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=232, interpolation="BILINEAR"),
                CenterCrop(width=224, height=224),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                Transpose(axis=[2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=232, interpolation="BILINEAR"),
                CenterCrop(width=224, height=224),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class RegNetY_1_6GF(ModelBase):
    info = ModelInfo(name="RegNetY_1_6GF", dataset=DatasetType.imagenet, evaluation=EvaluationType.image_classification)

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=232, interpolation="BILINEAR"),
                CenterCrop(width=224, height=224),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                Transpose(axis=[2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=232, interpolation="BILINEAR"),
                CenterCrop(width=224, height=224),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class RegNetY_16GF(ModelBase):
    info = ModelInfo(name="RegNetY_16GF", dataset=DatasetType.imagenet, evaluation=EvaluationType.image_classification)

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=384, interpolation="BILINEAR"),
                CenterCrop(width=384, height=384),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                Transpose(axis=[2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=384, interpolation="BILINEAR"),
                CenterCrop(width=384, height=384),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class RegNetY_3_2GF(ModelBase):
    info = ModelInfo(name="RegNetY_3_2GF", dataset=DatasetType.imagenet, evaluation=EvaluationType.image_classification)

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=232, interpolation="BILINEAR"),
                CenterCrop(width=224, height=224),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                Transpose(axis=[2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=232, interpolation="BILINEAR"),
                CenterCrop(width=224, height=224),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class RegNetY_32GF(ModelBase):
    info = ModelInfo(name="RegNetY_32GF", dataset=DatasetType.imagenet, evaluation=EvaluationType.image_classification)

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=384, interpolation="BILINEAR"),
                CenterCrop(width=384, height=384),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                Transpose(axis=[2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=384, interpolation="BILINEAR"),
                CenterCrop(width=384, height=384),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class RegNetY_8GF(ModelBase):
    info = ModelInfo(name="RegNetY_8GF", dataset=DatasetType.imagenet, evaluation=EvaluationType.image_classification)

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=232, interpolation="BILINEAR"),
                CenterCrop(width=224, height=224),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                Transpose(axis=[2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=232, interpolation="BILINEAR"),
                CenterCrop(width=224, height=224),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing
