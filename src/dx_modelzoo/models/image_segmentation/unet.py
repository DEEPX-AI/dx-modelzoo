import cv2
import numpy as np
from torchvision.transforms import Compose

from dx_modelzoo.enums import DatasetType, EvaluationType
from dx_modelzoo.models import ModelBase, ModelInfo
from dx_modelzoo.preprocessing.convertcolor import ConvertColor
from dx_modelzoo.preprocessing.div import Div
from dx_modelzoo.preprocessing.resize import Resize


def unet_label_preprocessing(label):
    """Resize label to 256x256 using nearest interpolation."""
    return cv2.resize(label.astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST).astype(np.int64)


def unet_postprocessing(outputs):
    pred = outputs[0]  # [1, 256, 256, 3] NHWC
    pred = np.argmax(pred, axis=-1)  # [1, 256, 256]
    return pred


class UNet_MobileNetV2(ModelBase):
    """UNet with MobileNetV2 encoder for image segmentation.

    Input: [1, 256, 256, 3] (NHWC format)
    Output: [1, 256, 256, 3] (3-class segmentation logits)
    Dataset: Oxford-IIIT Pet (3 classes: foreground, background, boundary)
    """

    info = ModelInfo(
        name="UNet_MobileNetV2",
        dataset=DatasetType.oxford_pet,
        evaluation=EvaluationType.oxford_pet,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)
        self.evaluator.dataset.label_preprocessing = unet_label_preprocessing

    def preprocessing(self):
        return Compose(
            [
                Resize(width=256, height=256),
                ConvertColor("BGR2RGB"),
                Div(255),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(width=256, height=256),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return unet_postprocessing
