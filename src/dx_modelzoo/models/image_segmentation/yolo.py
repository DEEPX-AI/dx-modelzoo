from typing import List

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


def xywh2xyxy(x):
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def yolov_non_max_suppression(
    prediction: torch.Tensor, conf_thres=0.001, iou_thres=0.45, max_det=300, nm=32
) -> List[torch.Tensor]:
    nc = prediction.shape[2] - nm - 5
    xc = prediction[..., 4] > conf_thres

    max_wh = 7680
    max_nms = 30000
    bs = prediction.shape[0]
    mi = 5 + nc
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs

    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if not x.shape[0]:
            continue

        x[:, 5:] *= x[:, 4:5]
        box = xywh2xyxy(x[:, :4])
        mask_coeffs = x[:, mi:]

        conf, j = x[:, 5:mi].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float(), mask_coeffs), 1)[conf.view(-1) > conf_thres]

        n = x.shape[0]
        if not n:
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        c = x[:, 5:6] * max_wh
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        i = i[:max_det]
        output[xi] = x[i]

    return output


def yolov5_postprocessing(outputs):
    return yolov_non_max_suppression(outputs)


class YoloV5L_Seg(ModelBase):
    info = ModelInfo("YoloV5L_Seg", dataset=DatasetType.coco, evaluation=EvaluationType.instance_segmentation)

    def __init__(self, evaluator: EvaluatorBase) -> None:
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=640, pad_location="edge", pad_value=[114, 114, 114]),
                Div(255),
                ConvertColor("BGR2RGB"),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=640, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return yolov5_postprocessing


class YoloV5M_Seg(ModelBase):
    info = ModelInfo("YoloV5M_Seg", dataset=DatasetType.coco, evaluation=EvaluationType.instance_segmentation)

    def __init__(self, evaluator: EvaluatorBase) -> None:
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=640, pad_location="edge", pad_value=[114, 114, 114]),
                Div(255),
                ConvertColor("BGR2RGB"),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=640, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return yolov5_postprocessing


class YoloV5N_Seg(ModelBase):
    info = ModelInfo("YoloV5N_Seg", dataset=DatasetType.coco, evaluation=EvaluationType.instance_segmentation)

    def __init__(self, evaluator: EvaluatorBase) -> None:
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=640, pad_location="edge", pad_value=[114, 114, 114]),
                Div(255),
                ConvertColor("BGR2RGB"),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=640, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return yolov5_postprocessing


class YoloV5S_Seg(ModelBase):
    info = ModelInfo("YoloV5S_Seg", dataset=DatasetType.coco, evaluation=EvaluationType.instance_segmentation)

    def __init__(self, evaluator: EvaluatorBase) -> None:
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=640, pad_location="edge", pad_value=[114, 114, 114]),
                Div(255),
                ConvertColor("BGR2RGB"),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=640, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return yolov5_postprocessing


def yolov8_postprocessing(prediction, conf_thres=0.001, iou_thres=0.45, max_det=300):
    """prediction: (batch_size, num_classes + 4 + num_masks, num_proposals)"""
    # (batch, channels, proposals) -> (batch, proposals, channels)
    prediction = prediction.permute(0, 2, 1)

    bs = prediction.shape[0]
    nc = prediction.shape[2] - 36  # 4 (box) + 32 (mask)
    nm = 32  # number of masks
    mi = 4 + nc
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs

    for xi, x in enumerate(prediction):
        conf = x[:, 4:mi].max(1, keepdim=True)[0]
        x = x[conf.view(-1) > conf_thres]

        if not x.shape[0]:
            continue

        box = xywh2xyxy(x[:, :4])
        mask = x[:, mi:]

        conf, j = x[:, 4:mi].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float(), mask), 1)

        n = x.shape[0]
        if not n:
            continue

        c = x[:, 5:6] * 7680
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        i = i[:max_det]
        output[xi] = x[i]

    return output


class YoloV8M_Seg(ModelBase):
    info = ModelInfo("YoloV8M_Seg", dataset=DatasetType.coco, evaluation=EvaluationType.instance_segmentation)

    def __init__(self, evaluator: EvaluatorBase) -> None:
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=640, pad_location="edge", pad_value=[114, 114, 114]),
                Div(255),
                ConvertColor("BGR2RGB"),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=640, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return yolov8_postprocessing


class YoloV8N_Seg(ModelBase):
    info = ModelInfo("YoloV8N_Seg", dataset=DatasetType.coco, evaluation=EvaluationType.instance_segmentation)

    def __init__(self, evaluator: EvaluatorBase) -> None:
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=640, pad_location="edge", pad_value=[114, 114, 114]),
                Div(255),
                ConvertColor("BGR2RGB"),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=640, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return yolov8_postprocessing


class YoloV8S_Seg(ModelBase):
    info = ModelInfo("YoloV8S_Seg", dataset=DatasetType.coco, evaluation=EvaluationType.instance_segmentation)

    def __init__(self, evaluator: EvaluatorBase) -> None:
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=640, pad_location="edge", pad_value=[114, 114, 114]),
                Div(255),
                ConvertColor("BGR2RGB"),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=640, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return yolov8_postprocessing
