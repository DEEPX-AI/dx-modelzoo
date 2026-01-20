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


def postprocess_tpu(outputs):
    outputs = outputs[0][:, 1:]

    return topk_postprocessing([outputs])


class EfficientNetB1(ModelBase):
    info = ModelInfo(
        name="EfficientNetB1",
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
                Resize(mode="torchvision", size=256, interpolation="BICUBIC"),
                CenterCrop(240, 240),
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
                CenterCrop(240, 240),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class EfficientNetB2(ModelBase):
    info = ModelInfo(
        name="EfficientNetB2",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
        raw_performance="80.61 95.31",
        q_lite_performance="79.19 94.55",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=288, interpolation="BICUBIC"),
                CenterCrop(288, 288),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=288, interpolation="BICUBIC"),
                CenterCrop(288, 288),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class EfficientNetV2S(ModelBase):
    info = ModelInfo(
        name="EfficientNetV2S",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
        raw_performance="84.24 96.87",
        q_lite_performance="80.5 95.20",
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
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=384, interpolation="BILINEAR"),
                CenterCrop(384, 384),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class EfficientNetV2M(ModelBase):
    info = ModelInfo(
        name="EfficientNetV2M",
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
                Resize(mode="torchvision", size=480, interpolation="BILINEAR"),
                CenterCrop(480, 480),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=480, interpolation="BILINEAR"),
                CenterCrop(480, 480),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class EfficientNetV2L(ModelBase):
    info = ModelInfo(
        name="EfficientNetV2L",
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
                Resize(mode="torchvision", size=480, interpolation="BICUBIC"),
                CenterCrop(480, 480),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=480, interpolation="BICUBIC"),
                CenterCrop(480, 480),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class EfficientNetLite0(ModelBase):
    info = ModelInfo(
        name="EfficientNetLite1",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
        # raw_performance="70.95 90.20",
        # q_lite_performance="71.09 90.18",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=256, interpolation="BILINEAR"),  # 224 * 1/0.875
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
                Resize(mode="torchvision", size=256, interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class EfficientNetLite1(ModelBase):
    info = ModelInfo(
        name="EfficientNetLite1",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
        raw_performance="70.95 90.20",
        q_lite_performance="71.09 90.18",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=274, interpolation="BILINEAR"),
                CenterCrop(240, 240),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=274, interpolation="BILINEAR"),
                CenterCrop(240, 240),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class EfficientNetLite2(ModelBase):
    info = ModelInfo(
        name="EfficientNetLite2",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
        raw_performance="",
        q_lite_performance="71.2 90.4",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=297, interpolation="BILINEAR"),
                CenterCrop(260, 260),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=297, interpolation="BILINEAR"),
                CenterCrop(260, 260),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class EfficientNetLite3(ModelBase):
    # NOTE: Official Input size is 280, but using 300 in this implementation
    # Resize: 320 -> CenterCrop: 280
    info = ModelInfo(
        name="EfficientNetLite3",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
        raw_performance="",
        q_lite_performance="78.4 94.1",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=343, interpolation="BILINEAR"),
                CenterCrop(300, 300),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=343, interpolation="BILINEAR"),
                CenterCrop(300, 300),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class EfficientNetLite4(ModelBase):
    # NOTE: Official Input size is 300, but using 380 in this implementation
    # Resize: 343 -> CenterCrop: 300
    info = ModelInfo(
        name="EfficientNetLite4",
        dataset=DatasetType.imagenet,
        evaluation=EvaluationType.image_classification,
        raw_performance="",
        q_lite_performance="77.8 93.9",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=434, interpolation="BILINEAR"),
                CenterCrop(380, 380),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        """NPU preprocessing for EfficientNet-Lite1.

        Same geometric transformations but without normalization
        (handled by NPU hardware).
        """
        return Compose(
            [
                Resize(mode="torchvision", size=434, interpolation="BILINEAR"),
                CenterCrop(380, 380),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return topk_postprocessing


class EfficientNetEdgeTPULarge(ModelBase):
    info = ModelInfo(
        name="EfficientNetEdgeTPULarge",
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
                Resize(mode="torchvision", size=343, interpolation="BILINEAR"),
                CenterCrop(300, 300),
                ConvertColor("BGR2RGB"),
                Normalize([127.0, 127.0, 127.0], [128.0, 128.0, 128.0]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=343, interpolation="BILINEAR"),
                CenterCrop(300, 300),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return postprocess_tpu


class EfficientNetEdgeTPUMedium(ModelBase):
    info = ModelInfo(
        name="EfficientNetEdgeTPUMedium",
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
                Resize(mode="torchvision", size=274, interpolation="BILINEAR"),
                CenterCrop(240, 240),
                ConvertColor("BGR2RGB"),
                Normalize([127.0, 127.0, 127.0], [128.0, 128.0, 128.0]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=274, interpolation="BILINEAR"),
                CenterCrop(240, 240),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return postprocess_tpu


class EfficientNetEdgeTPUSmall(ModelBase):
    info = ModelInfo(
        name="EfficientNetEdgeTPUMedium",
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
                Resize(mode="torchvision", size=224, interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
                Normalize([127.0, 127.0, 127.0], [128.0, 128.0, 128.0]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="torchvision", size=256, interpolation="BILINEAR"),
                CenterCrop(224, 224),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return postprocess_tpu
