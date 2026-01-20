from torchvision.transforms import Compose

from dx_modelzoo.enums import DatasetType, EvaluationType
from dx_modelzoo.models import ModelBase, ModelInfo
from dx_modelzoo.preprocessing.centercrop import CenterCrop
from dx_modelzoo.preprocessing.convertcolor import ConvertColor
from dx_modelzoo.preprocessing.div import Div
from dx_modelzoo.preprocessing.normalize import Normalize
from dx_modelzoo.preprocessing.resize import Resize
from dx_modelzoo.preprocessing.transpose import Transpose

DEEP_LAB_V3_LABEL_PREPROCESSING = Compose(
    [Resize(mode="torchvision", size=512, interpolation="NEAREST"), CenterCrop(512, 512)]
)


def deeplab_postprocessing(inputs):
    return inputs[0]


class DeepLabV3MobilenetV2(ModelBase):
    info = ModelInfo(
        name="DeepLabV3MobilenetV2",
        dataset=DatasetType.voc_seg,
        evaluation=EvaluationType.segmentation,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)
        self.evaluator.dataset.label_preprocessing = self.label_preprocessing()

    def preprocessing(self):
        return Compose(
            [
                ConvertColor("BGR2RGB"),
                Resize(mode="torchvision", size=512, interpolation="BILINEAR"),
                CenterCrop(512, 512),
                Div(255),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                ConvertColor("BGR2RGB"),
                Resize(mode="torchvision", size=512, interpolation="BILINEAR"),
                CenterCrop(512, 512),
            ]
        )

    def label_preprocessing(self):
        return DEEP_LAB_V3_LABEL_PREPROCESSING

    def postprocessing(self):
        return deeplab_postprocessing


class DeepLabV3PlusMobilenet(ModelBase):
    info = ModelInfo(
        name="DeepLabV3PlusMobilenet",
        dataset=DatasetType.voc_seg,
        evaluation=EvaluationType.segmentation,
        raw_performance="70.80",
        q_lite_performance="68.22",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)
        self.evaluator.dataset.label_preprocessing = self.label_preprocessing()

    def preprocessing(self):
        return Compose(
            [
                ConvertColor("BGR2RGB"),
                Resize(mode="torchvision", size=512, interpolation="BILINEAR"),
                CenterCrop(512, 512),
                Div(255),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                ConvertColor("BGR2RGB"),
                Resize(mode="torchvision", size=512, interpolation="BILINEAR"),
                CenterCrop(512, 512),
            ]
        )

    def label_preprocessing(self):
        return DEEP_LAB_V3_LABEL_PREPROCESSING

    def postprocessing(self):
        return deeplab_postprocessing


class DeepLabV3PlusResnet(ModelBase):
    info = ModelInfo(
        name="DeepLabV3PlusResnet",
        dataset=DatasetType.voc_seg,
        evaluation=EvaluationType.segmentation,
        raw_performance="75.13",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)
        self.evaluator.dataset.label_preprocessing = self.label_preprocessing()

    def preprocessing(self):
        return Compose(
            [
                ConvertColor("BGR2RGB"),
                Resize(mode="torchvision", size=512, interpolation="BILINEAR"),
                CenterCrop(512, 512),
                Div(255),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                ConvertColor("BGR2RGB"),
                Resize(mode="torchvision", size=512, interpolation="BILINEAR"),
                CenterCrop(512, 512),
            ]
        )

    def label_preprocessing(self):
        return DEEP_LAB_V3_LABEL_PREPROCESSING

    def postprocessing(self):
        return deeplab_postprocessing


class DeepLabV3PlusResNet50(ModelBase):
    info = ModelInfo(
        name="DeepLabV3PlusResNet50",
        dataset=DatasetType.voc_seg,
        evaluation=EvaluationType.segmentation,
        raw_performance="75.15",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)
        self.evaluator.dataset.label_preprocessing = self.label_preprocessing()

    def preprocessing(self):
        return Compose(
            [
                ConvertColor("BGR2RGB"),
                Resize(mode="torchvision", size=512, interpolation="BILINEAR"),
                CenterCrop(512, 512),
                Div(255),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                ConvertColor("BGR2RGB"),
                Resize(mode="torchvision", size=512, interpolation="BILINEAR"),
                CenterCrop(512, 512),
            ]
        )

    def label_preprocessing(self):
        return DEEP_LAB_V3_LABEL_PREPROCESSING

    def postprocessing(self):
        return deeplab_postprocessing


class DeepLabV3PlusResNet101(ModelBase):
    info = ModelInfo(
        name="DeepLabV3PlusResNet101",
        dataset=DatasetType.voc_seg,
        evaluation=EvaluationType.segmentation,
        raw_performance="76.10",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)
        self.evaluator.dataset.label_preprocessing = self.label_preprocessing()

    def preprocessing(self):
        return Compose(
            [
                ConvertColor("BGR2RGB"),
                Resize(mode="torchvision", size=512, interpolation="BILINEAR"),
                CenterCrop(512, 512),
                Div(255),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                ConvertColor("BGR2RGB"),
                Resize(mode="torchvision", size=512, interpolation="BILINEAR"),
                CenterCrop(512, 512),
            ]
        )

    def label_preprocessing(self):
        return DEEP_LAB_V3_LABEL_PREPROCESSING

    def postprocessing(self):
        return deeplab_postprocessing


class DeepLabV3PlusDRN(ModelBase):
    info = ModelInfo(
        name="DeepLabV3PlusDRN",
        dataset=DatasetType.voc_seg,
        evaluation=EvaluationType.segmentation,
        raw_performance="78.04",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)
        self.evaluator.dataset.label_preprocessing = self.label_preprocessing()

    def preprocessing(self):
        return Compose(
            [
                ConvertColor("BGR2RGB"),
                Resize(mode="torchvision", size=512, interpolation="BILINEAR"),
                CenterCrop(512, 512),
                Div(255),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                ConvertColor("BGR2RGB"),
                Resize(mode="torchvision", size=512, interpolation="BILINEAR"),
                CenterCrop(512, 512),
            ]
        )

    def label_preprocessing(self):
        return DEEP_LAB_V3_LABEL_PREPROCESSING

    def postprocessing(self):
        return deeplab_postprocessing


class DeepLabV3PlusMobileNetV2(ModelBase):
    info = ModelInfo(
        name="DeepLabV3PlusMobileNetV2",
        dataset=DatasetType.voc_seg,
        evaluation=EvaluationType.segmentation,
        raw_performance="70.81",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)
        self.evaluator.dataset.label_preprocessing = self.label_preprocessing()

    def preprocessing(self):
        return Compose(
            [
                ConvertColor("BGR2RGB"),
                Resize(mode="torchvision", size=512, interpolation="BILINEAR"),
                CenterCrop(512, 512),
                Div(255),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                ConvertColor("BGR2RGB"),
                Resize(mode="torchvision", size=512, interpolation="BILINEAR"),
                CenterCrop(512, 512),
            ]
        )

    def label_preprocessing(self):
        return DEEP_LAB_V3_LABEL_PREPROCESSING

    def postprocessing(self):
        return deeplab_postprocessing


DEEP_LAB_V3_LABEL_PREPROCESSING_2 = Compose(
    [Resize(mode="torchvision", size=513, interpolation="NEAREST"), CenterCrop(513, 513)]
)


class DeepLabV3MobileNetV2_Sim(ModelBase):
    info = ModelInfo(
        name="DeepLabV3MobileNetV2_Sim",
        dataset=DatasetType.voc_seg,
        evaluation=EvaluationType.segmentation,
        raw_performance="",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)
        self.evaluator.dataset.label_preprocessing = self.label_preprocessing()

    def preprocessing(self):
        return Compose(
            [
                ConvertColor("BGR2RGB"),
                Resize(mode="torchvision", size=513, interpolation="NEAREST"),
                CenterCrop(513, 513),
                Div(255),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                ConvertColor("BGR2RGB"),
                Resize(mode="torchvision", size=513, interpolation="BILINEAR"),
            ]
        )

    def label_preprocessing(self):
        return DEEP_LAB_V3_LABEL_PREPROCESSING_2

    def postprocessing(self):
        return deeplab_postprocessing
