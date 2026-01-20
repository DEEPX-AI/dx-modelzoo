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


class FastViT_MA36(ModelBase):
    info = ModelInfo(
        name="FastViT_MA36",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=256, interpolation="BICUBIC"),
                CenterCrop(256, 256),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=256, interpolation="BICUBIC"),
                CenterCrop(256, 256),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class FastViT_S12(ModelBase):
    info = ModelInfo(
        name="FastViT_S12",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=256, interpolation="BICUBIC"),
                CenterCrop(256, 256),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=256, interpolation="BICUBIC"),
                CenterCrop(256, 256),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class FastViT_SA12(ModelBase):
    info = ModelInfo(
        name="FastViT_SA12",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=256, interpolation="BICUBIC"),
                CenterCrop(256, 256),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=256, interpolation="BICUBIC"),
                CenterCrop(256, 256),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class FastViT_SA24(ModelBase):
    info = ModelInfo(
        name="FastViT_SA24",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=256, interpolation="BICUBIC"),
                CenterCrop(256, 256),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=256, interpolation="BICUBIC"),
                CenterCrop(256, 256),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class FastViT_SA36(ModelBase):
    info = ModelInfo(
        name="FastViT_SA36",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=256, interpolation="BICUBIC"),
                CenterCrop(256, 256),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=256, interpolation="BICUBIC"),
                CenterCrop(256, 256),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class FastViT_T12(ModelBase):
    info = ModelInfo(
        name="FastViT_T12",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=256, interpolation="BICUBIC"),
                CenterCrop(256, 256),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=256, interpolation="BICUBIC"),
                CenterCrop(256, 256),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class FastViT_T8(ModelBase):
    info = ModelInfo(
        name="FastViT_T8",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=256, interpolation="BICUBIC"),
                CenterCrop(256, 256),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=256, interpolation="BICUBIC"),
                CenterCrop(256, 256),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing
