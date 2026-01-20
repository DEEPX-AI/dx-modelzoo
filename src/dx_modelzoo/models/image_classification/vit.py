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


class ViT_Base_P16(ModelBase):
    info = ModelInfo(
        name="ViT_Base_P16",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=384, interpolation="BILINEAR"),
                CenterCrop(384, 384),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=384, interpolation="BILINEAR"),
                CenterCrop(384, 384),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                Transpose([2, 0, 1]),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class ViT_Base_P32(ModelBase):
    info = ModelInfo(
        name="ViT_Base_P16",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=224, interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=224, interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                Transpose([2, 0, 1]),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class ViT_Base_BN(ModelBase):
    info = ModelInfo(
        name="ViT_Base_BN",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=224, interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=224, interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                Transpose([2, 0, 1]),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class ViT_Base_P16_224(ModelBase):
    info = ModelInfo(
        name="ViT_Base_P16_224",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=224, interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=224, interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                Transpose([2, 0, 1]),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class ViT_Base_P16_384(ModelBase):
    info = ModelInfo(
        name="ViT_Base_P16_384",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=384, interpolation="BILINEAR"),
                CenterCrop(384, 384),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=384, interpolation="BILINEAR"),
                CenterCrop(384, 384),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                Transpose([2, 0, 1]),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class ViT_Base_P32_384(ModelBase):
    info = ModelInfo(
        name="ViT_Base_P32_384",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=384, interpolation="BILINEAR"),
                CenterCrop(384, 384),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=384, interpolation="BILINEAR"),
                CenterCrop(384, 384),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                Transpose([2, 0, 1]),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class ViT_Large_P32(ModelBase):
    info = ModelInfo(
        name="ViT_Large_P32",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=224, interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=224, interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                Transpose([2, 0, 1]),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class ViT_Small_P16_224(ModelBase):
    info = ModelInfo(
        name="ViT_Small_P16_224",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=224, interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=224, interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                Transpose([2, 0, 1]),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class ViT_Tiny_P16_224(ModelBase):
    info = ModelInfo(
        name="ViT_Tiny_P16_224",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=224, interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=224, interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                Transpose([2, 0, 1]),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class BeitBasePatch16_224(ModelBase):
    info = ModelInfo(
        name="BeitBasePatch16",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=256, interpolation="BICUBIC"),
                CenterCrop(224, 224),
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
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class BeitBasePatch16_384(ModelBase):
    info = ModelInfo(
        name="BeitBasePatch16",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=384, interpolation="BICUBIC"),
                CenterCrop(384, 384),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=512, interpolation="BICUBIC"),
                CenterCrop(384, 384),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class BeitLargePatch16_224(ModelBase):
    info = ModelInfo(
        name="BeitLargePatch16",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=256, interpolation="BICUBIC"),
                CenterCrop(224, 224),
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
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class BeitLargePatch16_384(ModelBase):
    info = ModelInfo(
        name="BeitLargePatch16",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=384, interpolation="BICUBIC"),
                CenterCrop(384, 384),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=512, interpolation="BICUBIC"),
                CenterCrop(384, 384),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class DeiTBase_224(ModelBase):
    info = ModelInfo(
        name="DeiTBase",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=256, interpolation="BICUBIC"),
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
                Resize(mode="torchvision", size=256, interpolation="BICUBIC"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class DeiTBaseDistilled_224(ModelBase):
    info = ModelInfo(
        name="DeiTBaseDistilled",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=256, interpolation="BICUBIC"),
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
                Resize(mode="torchvision", size=256, interpolation="BICUBIC"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class DeiTBase_384(ModelBase):
    info = ModelInfo(
        name="DeiTBase",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=384, interpolation="BICUBIC"),
                CenterCrop(384, 384),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=512, interpolation="BICUBIC"),
                CenterCrop(384, 384),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class DeiTBaseDistilled_384(ModelBase):
    info = ModelInfo(
        name="DeiTBaseDistilled",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=438, interpolation="BICUBIC"),
                CenterCrop(384, 384),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=438, interpolation="BICUBIC"),
                CenterCrop(384, 384),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class DeiTSmall_224(ModelBase):
    info = ModelInfo(
        name="DeiTSmall",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=256, interpolation="BICUBIC"),
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
                Resize(mode="torchvision", size=256, interpolation="BICUBIC"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class DeiTSmallDistilled_224(ModelBase):
    info = ModelInfo(
        name="DeiTSmallDistilled",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=256, interpolation="BICUBIC"),
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
                Resize(mode="torchvision", size=256, interpolation="BICUBIC"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class DeiTTiny_224(ModelBase):
    info = ModelInfo(
        name="DeiTTiny",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=256, interpolation="BICUBIC"),
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
                Resize(mode="torchvision", size=256, interpolation="BICUBIC"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class DeiTTinyDistilled_224(ModelBase):
    info = ModelInfo(
        name="DeiTTinyDistilled",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=256, interpolation="BICUBIC"),
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
                Resize(mode="torchvision", size=256, interpolation="BICUBIC"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing
