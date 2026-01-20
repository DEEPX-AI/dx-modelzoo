import cv2
import numpy as np
from torchvision.transforms import Compose

from dx_modelzoo.enums import DatasetType, EvaluationType
from dx_modelzoo.models import ModelBase, ModelInfo
from dx_modelzoo.preprocessing.centercrop import CenterCrop
from dx_modelzoo.preprocessing.convertcolor import ConvertColor
from dx_modelzoo.preprocessing.div import Div
from dx_modelzoo.preprocessing.normalize import Normalize
from dx_modelzoo.preprocessing.resize import Resize
from dx_modelzoo.preprocessing.transpose import Transpose


def fcn_postprocessing(outputs):
    pred = outputs[0]  # [B, C, H, W]
    pred = np.argmax(pred, axis=1)  # [B, H, W]
    return pred


FCN_LABLE_PREPROCESSING = Compose(
    [
        Resize(mode="torchvision", size=512, interpolation="BILINEAR"),
        CenterCrop(width=512, height=512),
    ]
)


class FCN8_ResNet50(ModelBase):
    info = ModelInfo(
        name="FCN8_ResNet50",
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
                CenterCrop(width=512, height=512),
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
                CenterCrop(width=512, height=512),
                Div(255),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                Transpose([2, 0, 1]),
            ]
        )

    def label_preprocessing(self):
        return FCN_LABLE_PREPROCESSING

    def postprocessing(self):
        return fcn_postprocessing


def fcn_postprocessing_2(outputs, target_size=(1024, 2048)):
    pred = outputs[0]  # [B, 1, H, W] 또는 [B, C, H, W]

    # squeeze channel dimension if exists
    if pred.ndim == 4 and pred.shape[1] == 1:
        pred = pred[:, 0, :, :]  # [B, H, W]
    elif pred.ndim == 4:
        pred = np.argmax(pred, axis=1)  # [B, H, W]

    resized = np.zeros((pred.shape[0], target_size[0], target_size[1]), dtype=np.int64)
    for i in range(pred.shape[0]):
        resized[i] = cv2.resize(
            pred[i].astype(np.uint8),
            (target_size[1], target_size[0]),  # cv2는 (width, height)
            interpolation=cv2.INTER_NEAREST,
        )

    return resized


class FCN8_ResNet18(ModelBase):
    info = ModelInfo(
        name="FCN8_ResNet18",
        dataset=DatasetType.city,
        evaluation=EvaluationType.segmentation,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                ConvertColor("BGR2RGB"),
                Resize(width=1920, height=1024),
                Div(255),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                ConvertColor("BGR2RGB"),
                Resize(width=1920, height=1024),
            ]
        )

    def postprocessing(self):
        return fcn_postprocessing_2
