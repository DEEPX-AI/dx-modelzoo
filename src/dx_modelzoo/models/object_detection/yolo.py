import os
from typing import List, Literal

import numpy as np
import torch
import torchvision
from torchvision.transforms import Compose

from dx_modelzoo.enums import DatasetType, EvaluationType
from dx_modelzoo.models import ModelBase, ModelInfo
from dx_modelzoo.models.object_detection.nms import (
    non_maximum_suppression,
    non_maximum_suppression2,
    non_maximum_suppression_iou,
)
from dx_modelzoo.preprocessing.convertcolor import ConvertColor
from dx_modelzoo.preprocessing.div import Div
from dx_modelzoo.preprocessing.resize import Resize
from dx_modelzoo.preprocessing.transpose import Transpose


def get_threshold_from_env(param_name: str, default_value: float) -> float:
    """Get threshold value from environment variable or use default"""
    env_value = os.environ.get(param_name)
    if env_value is not None:
        try:
            return float(env_value)
        except ValueError:
            pass
    return default_value


def yolo_detection_postprocessing(outputs, anchors_type: Literal["x", "w6"] = "x"):
    """
    YOLOv7 Detect layer style postprocessing for outputs
    """
    anchors_map = {
        "x": [
            [[12, 16], [19, 36], [40, 28]],
            [[36, 75], [76, 55], [72, 146]],
            [[142, 110], [192, 243], [459, 401]],
        ],
        "w6": [
            [[19, 27], [44, 40], [38, 94]],
            [[96, 68], [86, 152], [180, 137]],
            [[140, 301], [303, 264], [238, 542]],
            [[436, 615], [739, 380], [925, 792]],
        ],
    }
    anchors = anchors_map[anchors_type]
    strides = [8, 16, 32, 64]

    # Auto-detect number of scales
    num_scales = len(outputs)
    anchors = anchors[:num_scales]
    strides = strides[:num_scales]

    all_predictions = []

    if len(outputs[0].shape) == 3 and outputs[0].shape[-1] == 85:
        return non_maximum_suppression(outputs, multi_label=True)

    for scale_idx, output in enumerate(outputs):

        if isinstance(output, np.ndarray):
            output = torch.from_numpy(output).float()

        batch, num_anchors, h, w, num_props = output.shape

        # Create grid (matches _make_grid)
        grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        grid = torch.stack((grid_x, grid_y), dim=2).view(1, 1, h, w, 2).float()

        stride = strides[scale_idx]
        anchor_grid = torch.tensor(anchors[scale_idx]).float().view(1, num_anchors, 1, 1, 2)

        # Apply sigmoid (as in: y = x[i].sigmoid())
        y = torch.sigmoid(output)

        # Decode using YOLOv7 ONNX export formula
        xy = y[..., 0:2]
        wh = y[..., 2:4]

        # xy: xy * (2. * stride) + (stride * (grid - 0.5))
        #   = (xy * 2. - 0.5 + grid) * stride  (mathematically equivalent)
        xy_decoded = (xy * 2.0 - 0.5 + grid) * stride

        # wh: wh ** 2 * (4 * anchor_grid)
        #   = (wh * 2) ** 2 * anchor_grid  (mathematically equivalent)
        wh_decoded = (wh * 2.0) ** 2 * anchor_grid

        # Objectness and class scores (already sigmoid applied)
        obj_conf = y[..., 4:5]
        cls_conf = y[..., 5:]

        # Combine: [cx, cy, w, h, obj_conf, cls_conf...]
        pred = torch.cat([xy_decoded, wh_decoded, obj_conf, cls_conf], dim=-1)

        # Reshape: (1, 3, h, w, 85) -> (1, 3*h*w, 85)
        all_predictions.append(pred.view(batch, -1, num_props))

    # Concatenate all scales
    detections = torch.cat(all_predictions, dim=1)

    return non_maximum_suppression_iou(
        detections,
        conf_thres=0.001,
        iou_thres=0.7,
        multi_label=True,
        iou_type="iou",
    )


def yolo_postprocessing(outputs):
    return non_maximum_suppression(outputs, multi_label=True)


def yolov3_postprocessing(outputs):
    data = outputs[0]
    x1 = data[:, 0]
    y1 = data[:, 1]
    x2 = data[:, 2]
    y2 = data[:, 3]

    boxes = np.stack([x1, y1, x2, y2], axis=1)
    outputs = np.concatenate([boxes, data[:, 4:]], axis=1)[np.newaxis, ...]

    return non_maximum_suppression(outputs, multi_label=True)


class YoloV3(ModelBase):
    info = ModelInfo(
        name="YoloV3",
        dataset=DatasetType.coco,
        evaluation=EvaluationType.coco,
        raw_performance="46.65 66.05",
        q_lite_performance="46.41 65.89",
    )

    def __init__(self, evaluator):
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
        return yolo_postprocessing


class YoloV3_416(ModelBase):
    info = ModelInfo(
        name="YoloV3_416",
        dataset=DatasetType.coco,
        evaluation=EvaluationType.coco,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=416, pad_location="edge", pad_value=[114, 114, 114]),
                Div(255),
                ConvertColor("BGR2RGB"),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=416, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return yolo_postprocessing


class YoloV3_Tiny(ModelBase):
    info = ModelInfo(
        name="YoloV3_Tiny",
        dataset=DatasetType.coco,
        evaluation=EvaluationType.coco,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)
        self.evaluator.use_padding = False

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=416, pad_location="back", pad_value=[128, 128, 128]),
                Div(255),
                ConvertColor("BGR2RGB"),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=416, pad_location="back", pad_value=[128, 128, 128]),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return yolov3_postprocessing


class YoloV5N(ModelBase):
    info = ModelInfo(
        name="YoloV5N",
        dataset=DatasetType.coco,
        evaluation=EvaluationType.coco,
        raw_performance="28.08, 46.13",
        q_lite_performance="27.00, 44.79",
    )

    def __init__(self, evaluator):
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
        return yolo_postprocessing


class YoloV5S(ModelBase):
    info = ModelInfo(
        name="YoloV5S",
        dataset=DatasetType.coco,
        evaluation=EvaluationType.coco,
        raw_performance="37.45 57.08",
        q_lite_performance="36.91 56.53",
    )

    def __init__(self, evaluator):
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
        return yolo_postprocessing


class YoloV5S_320(ModelBase):
    info = ModelInfo(
        name="YoloV5S_320",
        dataset=DatasetType.coco,
        evaluation=EvaluationType.coco,
        raw_performance="",
        q_lite_performance="",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=320, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor("BGR2RGB"),
                Div(255),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=320, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return yolo_postprocessing


class YoloV5M(ModelBase):
    info = ModelInfo(
        name="YoloV5M",
        dataset=DatasetType.coco,
        evaluation=EvaluationType.coco,
        raw_performance="45.08 64.14",
        q_lite_performance="44.67 63.95",
    )

    def __init__(self, evaluator):
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
        return yolo_postprocessing


class YoloV5L(ModelBase):
    info = ModelInfo(
        name="YoloV5L",
        dataset=DatasetType.coco,
        evaluation=EvaluationType.coco,
        raw_performance="48.74 67.16",
        q_lite_performance="48.34 67.10",
    )

    def __init__(self, evaluator):
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
        return yolo_postprocessing


class YoloV6N(ModelBase):
    info = ModelInfo(
        name="YoloV6N",
        dataset=DatasetType.coco,
        evaluation=EvaluationType.coco,
        raw_performance="35.0 / 36.3",  # ver 0.1.0 and 0.2.1 Based on GitHub
        q_lite_performance="35.5 / 35.1",  # ver 0.1.0 and 0.2.1
    )

    def __init__(self, evaluator):
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
        return yolo_postprocessing


class YoloV7(ModelBase):
    info = ModelInfo(
        name="YoloV7",
        dataset=DatasetType.coco,
        evaluation=EvaluationType.coco,
    )

    def __init__(self, evaluator):
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
        return lambda x: yolo_detection_postprocessing(x, anchors_type="x")


class YoloV7_wo_decoding(ModelBase):
    info = ModelInfo(
        name="YoloV7_wo_decoding",
        dataset=DatasetType.coco,
        evaluation=EvaluationType.coco,
    )

    def __init__(self, evaluator):
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
        return lambda x: yolo_detection_postprocessing(x, anchors_type="x")


class YoloV7E6(ModelBase):
    info = ModelInfo(
        name="YoloV7E6",
        dataset=DatasetType.coco,
        evaluation=EvaluationType.coco,
        raw_performance="55.22 72.97",
        q_lite_performance="55.15 72.90",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=1280, pad_location="edge", pad_value=[114, 114, 114]),
                Div(255),
                ConvertColor("BGR2RGB"),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=1280, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return yolo_postprocessing


class YoloV7_X(ModelBase):
    info = ModelInfo(
        name="YoloV7_X",
        dataset=DatasetType.coco,
        evaluation=EvaluationType.coco,
        raw_performance="",
        q_lite_performance="",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=640, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor("BGR2RGB"),
                Div(255),
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
        return lambda x: yolo_detection_postprocessing(x, anchors_type="x")


class YoloV7_W6(ModelBase):
    info = ModelInfo(
        name="YoloV7_W6",
        dataset=DatasetType.coco,
        evaluation=EvaluationType.coco,
        raw_performance="",
        q_lite_performance="",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=1280, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor("BGR2RGB"),
                Div(255),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=1280, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return lambda x: yolo_detection_postprocessing(x, anchors_type="w6")


class YoloV7_W6_wo_decoding(ModelBase):
    info = ModelInfo(
        name="YoloV7_W6_wo_decoding",
        dataset=DatasetType.coco,
        evaluation=EvaluationType.coco,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=1280, pad_location="edge", pad_value=[114, 114, 114]),
                Div(255),
                ConvertColor("BGR2RGB"),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=1280, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return lambda x: yolo_detection_postprocessing(x, anchors_type="w6")


class YoloV7Tiny(ModelBase):
    info = ModelInfo(
        name="YoloV7Tiny",
        dataset=DatasetType.coco,
        evaluation=EvaluationType.coco,
        raw_performance="37.29 55.42",
        q_lite_performance="37.08 55.21",
    )

    def __init__(self, evaluator):
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
        return yolo_postprocessing


class YoloXS(ModelBase):
    info = ModelInfo(
        name="YoloXS",
        dataset=DatasetType.coco,
        evaluation=EvaluationType.coco,
        raw_performance="40.29 59.31",
        q_lite_performance="39.90 59.01",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

        output_strides = [8, 16, 32]
        input_size = 640
        grids = []
        strides = []
        for stride in output_strides:
            output_size = input_size // stride
            arange = torch.arange(output_size)
            yv, xv = torch.meshgrid(arange, arange, indexing="ij")
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))
        self.grids = torch.cat(grids, dim=1).float()
        self.strides = torch.cat(strides, dim=1).float()

    def preprocessing(self):
        return Compose(
            [Resize(mode="pad", size=640, pad_location="edge", pad_value=[114, 114, 114]), Transpose([2, 0, 1])]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=640, pad_location="edge", pad_value=[114, 114, 114]),
            ]
        )

    def postprocessing(self):
        def _yolox_postprocessing(outputs: List[np.ndarray]):
            outputs = outputs[0]

            outputs = torch.from_numpy(outputs)
            outputs = torch.cat(
                [
                    (outputs[..., 0:2] + self.grids) * self.strides,
                    torch.exp(outputs[..., 2:4]) * self.strides,
                    outputs[..., 4:],
                ],
                dim=-1,
            )
            return non_maximum_suppression(outputs)

        return _yolox_postprocessing


class YoloXTiny(ModelBase):
    info = ModelInfo(
        name="YoloXTiny",
        dataset=DatasetType.coco,
        evaluation=EvaluationType.coco,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

        output_strides = [8, 16, 32]
        input_size = 416
        grids = []
        strides = []
        for stride in output_strides:
            output_size = input_size // stride
            arange = torch.arange(output_size)
            yv, xv = torch.meshgrid(arange, arange, indexing="ij")
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))
        self.grids = torch.cat(grids, dim=1).float()
        self.strides = torch.cat(strides, dim=1).float()

    def preprocessing(self):
        return Compose(
            [Resize(mode="pad", size=416, pad_location="edge", pad_value=[114, 114, 114]), Transpose([2, 0, 1])]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=416, pad_location="edge", pad_value=[114, 114, 114]),
            ]
        )

    def postprocessing(self):
        def _yolox_postprocessing(outputs: List[np.ndarray]):
            outputs = outputs[0]

            outputs = torch.from_numpy(outputs)
            outputs = torch.cat(
                [
                    (outputs[..., 0:2] + self.grids) * self.strides,
                    torch.exp(outputs[..., 2:4]) * self.strides,
                    outputs[..., 4:],
                ],
                dim=-1,
            )
            return non_maximum_suppression(outputs)

        return _yolox_postprocessing


class YoloXSWideLeaky(ModelBase):
    info = ModelInfo(
        name="YoloXSWideLeaky",
        dataset=DatasetType.coco,
        evaluation=EvaluationType.coco,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

        output_strides = [8, 16, 32]
        input_size = 640
        grids = []
        strides = []
        for stride in output_strides:
            output_size = input_size // stride
            arange = torch.arange(output_size)
            yv, xv = torch.meshgrid(arange, arange, indexing="ij")
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))
        self.grids = torch.cat(grids, dim=1).float()
        self.strides = torch.cat(strides, dim=1).float()

    def preprocessing(self):
        return Compose(
            [Resize(mode="pad", size=640, pad_location="edge", pad_value=[114, 114, 114]), Transpose([2, 0, 1])]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=640, pad_location="edge", pad_value=[114, 114, 114]),
            ]
        )

    def postprocessing(self):
        def _yolox_postprocessing(outputs: List[np.ndarray]):
            outputs = outputs[0]

            outputs = torch.from_numpy(outputs)
            outputs = torch.cat(
                [
                    (outputs[..., 0:2] + self.grids) * self.strides,
                    torch.exp(outputs[..., 2:4]) * self.strides,
                    outputs[..., 4:],
                ],
                dim=-1,
            )
            return non_maximum_suppression(outputs)

        return _yolox_postprocessing


class YoloXSLeaky(ModelBase):
    info = ModelInfo(
        name="YoloXSLeaky",
        dataset=DatasetType.coco,
        evaluation=EvaluationType.coco,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

        output_strides = [8, 16, 32]
        input_size = 640
        grids = []
        strides = []
        for stride in output_strides:
            output_size = input_size // stride
            arange = torch.arange(output_size)
            yv, xv = torch.meshgrid(arange, arange, indexing="ij")
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))
        self.grids = torch.cat(grids, dim=1).float()
        self.strides = torch.cat(strides, dim=1).float()

    def preprocessing(self):
        return Compose(
            [Resize(mode="pad", size=640, pad_location="edge", pad_value=[114, 114, 114]), Transpose([2, 0, 1])]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=640, pad_location="edge", pad_value=[114, 114, 114]),
            ]
        )

    def postprocessing(self):
        def _yolox_postprocessing(outputs: List[np.ndarray]):
            outputs = outputs[0]

            outputs = torch.from_numpy(outputs)
            outputs = torch.cat(
                [
                    (outputs[..., 0:2] + self.grids) * self.strides,
                    torch.exp(outputs[..., 2:4]) * self.strides,
                    outputs[..., 4:],
                ],
                dim=-1,
            )
            return non_maximum_suppression(outputs)

        return _yolox_postprocessing


class YoloXLLeaky(ModelBase):
    info = ModelInfo(
        name="YoloXLLeaky",
        dataset=DatasetType.coco,
        evaluation=EvaluationType.coco,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

        output_strides = [8, 16, 32]
        input_size = 640
        grids = []
        strides = []
        for stride in output_strides:
            output_size = input_size // stride
            arange = torch.arange(output_size)
            yv, xv = torch.meshgrid(arange, arange, indexing="ij")
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))
        self.grids = torch.cat(grids, dim=1).float()
        self.strides = torch.cat(strides, dim=1).float()

    def preprocessing(self):
        return Compose(
            [Resize(mode="pad", size=640, pad_location="edge", pad_value=[114, 114, 114]), Transpose([2, 0, 1])]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=640, pad_location="edge", pad_value=[114, 114, 114]),
            ]
        )

    def postprocessing(self):
        def _yolox_postprocessing(outputs: List[np.ndarray]):
            outputs = outputs[0]

            outputs = torch.from_numpy(outputs)
            outputs = torch.cat(
                [
                    (outputs[..., 0:2] + self.grids) * self.strides,
                    torch.exp(outputs[..., 2:4]) * self.strides,
                    outputs[..., 4:],
                ],
                dim=-1,
            )
            return non_maximum_suppression(outputs)

        return _yolox_postprocessing


def yolov8_postprocessing(outputs: List[np.ndarray]):
    outputs = outputs[0]
    outputs = torch.from_numpy(outputs)
    outputs = outputs.transpose(1, 2)

    return non_maximum_suppression2(outputs, iou_thres=0.65)


class YoloV8X(ModelBase):
    info = ModelInfo(name="YoloV8X", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

    def __init__(self, evaluator):
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


class YoloV8N(ModelBase):
    info = ModelInfo(name="YoloV8N", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

    def __init__(self, evaluator):
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


class YoloV8S(ModelBase):
    info = ModelInfo(name="YoloV8S", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

    def __init__(self, evaluator):
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


class YoloV8M(ModelBase):
    info = ModelInfo(name="YoloV8M", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

    def __init__(self, evaluator):
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


class YoloV8L(ModelBase):
    info = ModelInfo(name="YoloV8L", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

    def __init__(self, evaluator):
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


def yolov9_postprocessing(outputs: List[np.ndarray]):
    outputs = outputs[0]
    outputs = torch.from_numpy(outputs)
    outputs = outputs.transpose(1, 2)

    return non_maximum_suppression2(outputs, iou_thres=0.7)


class YoloV9T(ModelBase):
    info = ModelInfo(
        name="YoloV9T",
        dataset=DatasetType.coco,
        evaluation=EvaluationType.coco,
    )

    def __init__(self, evaluator):
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
        return yolov9_postprocessing


class YoloV9S(ModelBase):
    info = ModelInfo(name="YoloV9S", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

    def __init__(self, evaluator):
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
        return yolov9_postprocessing


class YoloV9C(ModelBase):
    info = ModelInfo(name="YoloV9C", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

    def __init__(self, evaluator):
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
        return yolov9_postprocessing


def damoyolo_postprocessing(outputs: List[np.ndarray]):
    if not isinstance(outputs, list):
        outputs = [outputs]

    if len(outputs) == 1 and outputs[0].shape[-1] == 80:
        return torch.empty((0, 6), dtype=torch.float32)

    if len(outputs) == 2 and outputs[0].shape[-1] == 80:
        outputs = np.concatenate([outputs[1], outputs[0]], -1)
    elif len(outputs) > 1:
        outputs = np.concatenate(outputs, -1)
    else:
        outputs = outputs[0]

    if outputs.shape[-1] != 84:
        return torch.empty((0, 6), dtype=torch.float32)

    outputs = torch.from_numpy(outputs)
    return non_maximum_suppression2(outputs, conf_thres=0.005, iou_thres=0.7, cxcywh2xyxy_conversion=False)


class DamoYoloM(ModelBase):
    info = ModelInfo(name="DamoYoloM", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=640, pad_location="edge", pad_value=[114, 114, 114]),
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
        return damoyolo_postprocessing


class DamoYoloS(ModelBase):
    info = ModelInfo(name="DamoYoloM", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=640, pad_location="edge", pad_value=[114, 114, 114]),
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
        return damoyolo_postprocessing


class DamoYoloT(ModelBase):
    info = ModelInfo(name="DamoYoloM", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=640, pad_location="edge", pad_value=[114, 114, 114]),
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
        return damoyolo_postprocessing


def yolov11_postprocessing(outputs: List[np.ndarray]):
    outputs = outputs[0]
    outputs = torch.from_numpy(outputs)
    outputs = outputs.transpose(1, 2)

    return non_maximum_suppression2(outputs, iou_thres=0.65)


class YoloV11(ModelBase):
    info = ModelInfo(name="YoloV11", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

    def __init__(self, evaluator):
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
        return yolov11_postprocessing


def yolov10_postprocessing(outputs: List[np.ndarray]):    
    outputs = outputs[0]
    outputs = torch.from_numpy(outputs)

    return outputs[0]

class YoloV10B(ModelBase):
    info = ModelInfo(name="YoloV10B", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

    def __init__(self, evaluator):
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
        return yolov10_postprocessing


class YoloV10N(ModelBase):
    info = ModelInfo(name="YoloV10N", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

    def __init__(self, evaluator):
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
        return yolov10_postprocessing


class YoloV10S(ModelBase):
    info = ModelInfo(name="YoloV10S", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

    def __init__(self, evaluator):
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
        return yolov10_postprocessing


class YoloV10M(ModelBase):
    info = ModelInfo(name="YoloV10M", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

    def __init__(self, evaluator):
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
        return yolov10_postprocessing


class YoloV10L(ModelBase):
    info = ModelInfo(name="YoloV10L", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

    def __init__(self, evaluator):
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
        return yolov10_postprocessing


class YoloV10X(ModelBase):
    info = ModelInfo(name="YoloV10X", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

    def __init__(self, evaluator):
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
        return yolov10_postprocessing


class YoloV10N_PPU(ModelBase):
    info = ModelInfo(
        name="YoloV10N_PPU",
        dataset=DatasetType.coco,
        evaluation=EvaluationType.coco,
        raw_performance="",
        q_lite_performance="",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)
        self.evaluator.lazy_postprocessing = True
        self.evaluator.use_ppu = True

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
        return yolov10_postprocessing


def yolo26_postprocessing(outputs: List[np.ndarray]):
    outputs = outputs[0]
    outputs = torch.from_numpy(outputs)

    return outputs[0]


class YOLO26n(ModelBase):
    info = ModelInfo(name="YOLO26n", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

    def __init__(self, evaluator):
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
        return yolo26_postprocessing


class YOLO26s(ModelBase):
    info = ModelInfo(name="YOLO26s", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

    def __init__(self, evaluator):
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
        return yolo26_postprocessing


class YOLO26m(ModelBase):
    info = ModelInfo(name="YOLO26m", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

    def __init__(self, evaluator):
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
        return yolo26_postprocessing


class YOLO26l(ModelBase):
    info = ModelInfo(name="YOLO26l", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

    def __init__(self, evaluator):
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
        return yolo26_postprocessing


class YOLO26x(ModelBase):
    info = ModelInfo(name="YOLO26x", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

    def __init__(self, evaluator):
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
        return yolo26_postprocessing
