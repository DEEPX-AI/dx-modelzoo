"""FastSAM (Fast Segment Anything) model.

YOLOv8-seg based zero-shot instance segmentation.
Single class ("everything") with 32 mask prototypes.
"""

import numpy as np
import torch
import torchvision
from torchvision.transforms import Compose

from dx_modelzoo.enums import DatasetType, EvaluationType
from dx_modelzoo.evaluator import EvaluatorBase
from dx_modelzoo.models import ModelBase, ModelInfo
from dx_modelzoo.preprocessing.convertcolor import ConvertColor
from dx_modelzoo.preprocessing.div import Div
from dx_modelzoo.preprocessing.resize import Resize
from dx_modelzoo.preprocessing.transpose import Transpose


def fastsam_postprocessing(prediction, conf_thres=0.001, iou_thres=0.7, max_det=300):
    """FastSAM postprocessing. Same as YOLOv8 seg but with 1 class."""
    prediction = prediction.permute(0, 2, 1)

    bs = prediction.shape[0]
    nc = prediction.shape[2] - 36  # 4 (box) + 32 (mask)
    nm = 32
    mi = 4 + nc
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs

    for xi, x in enumerate(prediction):
        conf = x[:, 4:mi].max(1, keepdim=True)[0]
        x = x[conf.view(-1) > conf_thres]

        if not x.shape[0]:
            continue

        box = _xywh2xyxy(x[:, :4])
        mask = x[:, mi:]

        conf, j = x[:, 4:mi].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float(), mask), 1)

        n = x.shape[0]
        if not n:
            continue

        x = x[x[:, 4].argsort(descending=True)[:30000]]
        boxes, scores = x[:, :4], x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        i = i[:max_det]
        output[xi] = x[i]

    return output


def _xywh2xyxy(x):
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


class FastSAM_S(ModelBase):
    """FastSAM-S: Fast Segment Anything Model (small)."""

    info = ModelInfo(
        "FastSAM_S",
        dataset=DatasetType.coco,
        evaluation=EvaluationType.zeroshot_instance_segmentation,
    )

    def __init__(self, evaluator: EvaluatorBase) -> None:
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=1024, pad_location="edge", pad_value=[114, 114, 114]),
                Div(255),
                ConvertColor("BGR2RGB"),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=1024, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return fastsam_postprocessing
