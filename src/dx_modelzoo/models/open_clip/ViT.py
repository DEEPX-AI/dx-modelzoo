from torchvision.transforms import Compose

from dx_modelzoo.enums import DatasetType, EvaluationType
from dx_modelzoo.models import ModelBase, ModelInfo
from dx_modelzoo.preprocessing.centercrop import CenterCrop
from dx_modelzoo.preprocessing.convertcolor import ConvertColor
from dx_modelzoo.preprocessing.div import Div
from dx_modelzoo.preprocessing.normalize import Normalize
from dx_modelzoo.preprocessing.resize import Resize
from dx_modelzoo.preprocessing.transpose import Transpose


class ViT_B_16_dfn2b(ModelBase):
    info = ModelInfo(
        name="ViT_B_16_dfn2b",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.zeroshot_classification,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(width=224, height=224),
                CenterCrop(height=224, width=224),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(width=224, height=224),
                CenterCrop(height=224, width=224),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
                Transpose([2, 0, 1]),
            ]
        )

    def postprocessing(self):
        def open_clip_postprocessing(output):
            return output[0]

        return open_clip_postprocessing


class ViT_B_16_quickgelu_metaclip_fullcc(ModelBase):
    info = ModelInfo(
        name="ViT_B_16_quickgelu_metaclip_fullcc",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.zeroshot_classification,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(width=224, height=224),
                CenterCrop(height=224, width=224),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(width=224, height=224),
                CenterCrop(height=224, width=224),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
                Transpose([2, 0, 1]),
            ]
        )

    def postprocessing(self):
        def open_clip_postprocessing(output):
            return output[0]

        return open_clip_postprocessing


class ViT_B_32_256_datacomp_s34b_b68k(ModelBase):
    info = ModelInfo(
        name="ViT_B_32_256_datacomp_s34b_b68k",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.zeroshot_classification,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(width=256, height=256),
                CenterCrop(height=256, width=256),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(width=256, height=256),
                CenterCrop(height=256, width=256),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
                Transpose([2, 0, 1]),
            ]
        )

    def postprocessing(self):
        def open_clip_postprocessing(output):
            return output[0]

        return open_clip_postprocessing


class ViT_L_14_CLIPA_datacomp1b(ModelBase):
    info = ModelInfo(
        name="ViT_L_14_CLIPA_datacomp1b",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.zeroshot_classification,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(width=224, height=224),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(width=224, height=224),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                Transpose([2, 0, 1]),
            ]
        )

    def postprocessing(self):
        def open_clip_postprocessing(output):
            return output[0]

        return open_clip_postprocessing


class ViT_L_14_datacomp_xl_s13b_b90k(ModelBase):
    info = ModelInfo(
        name="ViT_L_14_datacomp_xl_s13b_b90k",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.zeroshot_classification,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(width=224, height=224),
                CenterCrop(height=224, width=224),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(width=224, height=224),
                CenterCrop(height=224, width=224),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
                Transpose([2, 0, 1]),
            ]
        )

    def postprocessing(self):
        def open_clip_postprocessing(output):
            return output[0]

        return open_clip_postprocessing


class ViT_L_14_336_openai(ModelBase):
    info = ModelInfo(
        name="ViT_L_14_336_openai",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.zeroshot_classification,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(width=336, height=336),
                CenterCrop(height=336, width=336),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(width=336, height=336),
                CenterCrop(height=336, width=336),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
                Transpose([2, 0, 1]),
            ]
        )

    def postprocessing(self):
        def open_clip_postprocessing(output):
            return output[0]

        return open_clip_postprocessing


class ViT_L_14_CLIPA_336_datacomp1b(ModelBase):
    info = ModelInfo(
        name="ViT_L_14_CLIPA_336_datacomp1b",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.zeroshot_classification,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(width=336, height=336),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(width=336, height=336),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                Transpose([2, 0, 1]),
            ]
        )

    def postprocessing(self):
        def open_clip_postprocessing(output):
            return output[0]

        return open_clip_postprocessing


class ViT_L_14_quickgelu_dfn2b(ModelBase):
    info = ModelInfo(
        name="ViT_L_14_quickgelu_dfn2b",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.zeroshot_classification,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(width=224, height=224),
                CenterCrop(height=224, width=224),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(width=224, height=224),
                CenterCrop(height=224, width=224),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
                Transpose([2, 0, 1]),
            ]
        )

    def postprocessing(self):
        def open_clip_postprocessing(output):
            return output[0]

        return open_clip_postprocessing
