from dx_modelzoo.enums import DatasetType, EvaluationType, SessionType
from dx_modelzoo.models import ModelBase, ModelInfo
from dx_modelzoo.preprocessing import PreProcessingCompose

DEEP_LAB_V3_LABEL_PREPROCESSING = PreProcessingCompose(
    preprocessings=[
        {
            "resize": {
                "mode": "torchvision",
                "size": 512,
                "interpolation": "NEAREST",
            }
        },
        {"centercrop": {"width": 512, "height": 512}},
    ],
    session_type=SessionType.onnxruntime,
)


def deeplab_postprocessing(inputs):
    return inputs[0]


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
        return [
            {"convertColor": {"form": "BGR2RGB"}},
            {
                "resize": {
                    "mode": "torchvision",
                    "size": 512,
                    "interpolation": "BILINEAR",
                }
            },
            {"centercrop": {"width": 512, "height": 512}},
            {"div": {"x": 255}},
            {
                "normalize": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                }
            },
            {"transpose": {"axis": [2, 0, 1]}},
            {"expandDim": {"axis": 0}},
        ]

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
        return [
            {"convertColor": {"form": "BGR2RGB"}},
            {
                "resize": {
                    "mode": "torchvision",
                    "size": 512,
                    "interpolation": "BILINEAR",
                }
            },
            {"centercrop": {"width": 512, "height": 512}},
            {"div": {"x": 255}},
            {
                "normalize": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                }
            },
            {"transpose": {"axis": [2, 0, 1]}},
            {"expandDim": {"axis": 0}},
        ]

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
        return [
            {"convertColor": {"form": "BGR2RGB"}},
            {
                "resize": {
                    "mode": "torchvision",
                    "size": 512,
                    "interpolation": "BILINEAR",
                }
            },
            {"centercrop": {"width": 512, "height": 512}},
            {"div": {"x": 255}},
            {
                "normalize": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                }
            },
            {"transpose": {"axis": [2, 0, 1]}},
            {"expandDim": {"axis": 0}},
        ]

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
        return [
            {"convertColor": {"form": "BGR2RGB"}},
            {
                "resize": {
                    "mode": "torchvision",
                    "size": 512,
                    "interpolation": "BILINEAR",
                }
            },
            {"centercrop": {"width": 512, "height": 512}},
            {"div": {"x": 255}},
            {
                "normalize": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                }
            },
            {"transpose": {"axis": [2, 0, 1]}},
            {"expandDim": {"axis": 0}},
        ]

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
        return [
            {"convertColor": {"form": "BGR2RGB"}},
            {
                "resize": {
                    "mode": "torchvision",
                    "size": 512,
                    "interpolation": "BILINEAR",
                }
            },
            {"centercrop": {"width": 512, "height": 512}},
            {"div": {"x": 255}},
            {
                "normalize": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                }
            },
            {"transpose": {"axis": [2, 0, 1]}},
            {"expandDim": {"axis": 0}},
        ]

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
        return [
            {"convertColor": {"form": "BGR2RGB"}},
            {
                "resize": {
                    "mode": "torchvision",
                    "size": 512,
                    "interpolation": "BILINEAR",
                }
            },
            {"centercrop": {"width": 512, "height": 512}},
            {"div": {"x": 255}},
            {
                "normalize": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                }
            },
            {"transpose": {"axis": [2, 0, 1]}},
            {"expandDim": {"axis": 0}},
        ]

    def label_preprocessing(self):
        return DEEP_LAB_V3_LABEL_PREPROCESSING

    def postprocessing(self):
        return deeplab_postprocessing
