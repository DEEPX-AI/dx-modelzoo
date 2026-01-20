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


def segformer_postprocessing(outputs, target_size=(1024, 2048)):
    pred = outputs[0]  # [B, 1, H, W] or [B, C, H, W]

    if pred.ndim == 4 and pred.shape[1] == 1:
        pred = pred[:, 0, :, :]  # [B, H, W]
    elif pred.ndim == 4:
        pred = np.argmax(pred, axis=1)  # [B, H, W]

    resized = np.zeros((pred.shape[0], target_size[0], target_size[1]), dtype=np.int64)
    for i in range(pred.shape[0]):
        resized[i] = cv2.resize(
            pred[i].astype(np.uint8),
            (target_size[1], target_size[0]),  # (width, height)
            interpolation=cv2.INTER_NEAREST,
        )

    return resized


class SegFormer_b0_512x1024(ModelBase):
    info = ModelInfo(
        name="SegFormer_b0_512x1024",
        dataset=DatasetType.city,
        evaluation=EvaluationType.segmentation,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                ConvertColor("BGR2RGB"),
                Resize(width=1024, height=512),
                Div(255),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                ConvertColor("BGR2RGB"),
                Resize(width=1024, height=512),
            ]
        )

    def postprocessing(self):
        return segformer_postprocessing
