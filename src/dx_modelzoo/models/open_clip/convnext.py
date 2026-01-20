from torchvision.transforms import Compose

from dx_modelzoo.enums import DatasetType, EvaluationType
from dx_modelzoo.models import ModelBase, ModelInfo
from dx_modelzoo.preprocessing.centercrop import CenterCrop
from dx_modelzoo.preprocessing.convertcolor import ConvertColor
from dx_modelzoo.preprocessing.div import Div
from dx_modelzoo.preprocessing.normalize import Normalize
from dx_modelzoo.preprocessing.resize import Resize
from dx_modelzoo.preprocessing.transpose import Transpose


class ConvNextBase_w_laion2b_s13b_b82k_augreg(ModelBase):
    info = ModelInfo(
        name="ConvNextBase_w_laion2b_s13b_b82k_augreg",
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
            ]
        )

    def postprocessing(self):
        def open_clip_postprocessing(output):
            return output[0]

        return open_clip_postprocessing


class ConvNextBase_w_320_laion_aesthetic_s13b_b82k(ModelBase):
    info = ModelInfo(
        name="ConvNextBase_w_320_laion_aesthetic_s13b_b82k",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.zeroshot_classification,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(width=320, height=320),
                CenterCrop(height=320, width=320),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(width=320, height=320),
                CenterCrop(height=320, width=320),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        def open_clip_postprocessing(output):
            return output[0]

        return open_clip_postprocessing


class ConvNextLarge_d_laion2b_s26b_b102k_augreg(ModelBase):
    info = ModelInfo(
        name="ConvNextLarge_d_laion2b_s26b_b102k_augreg",
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
            ]
        )

    def postprocessing(self):
        def open_clip_postprocessing(output):
            return output[0]

        return open_clip_postprocessing


class ConvNextLarge_w_320_laion2b_s29b_b131k_ft_soup(ModelBase):
    info = ModelInfo(
        name="ConvNextLarge_w_320_laion2b_s29b_b131k_ft_soup",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.zeroshot_classification,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(width=320, height=320),
                CenterCrop(height=320, width=320),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(width=320, height=320),
                CenterCrop(height=320, width=320),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        def open_clip_postprocessing(output):
            return output[0]

        return open_clip_postprocessing
