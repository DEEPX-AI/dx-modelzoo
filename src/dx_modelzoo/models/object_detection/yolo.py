import os
from typing import List, Literal

import numpy as np
import torch
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
        return yolo_postprocessing


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
    

class YoloV6N0_1_0(ModelBase):
    info = ModelInfo(
        name="YoloV6N0_1_0",
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
    

class YoloV6N0_2_1(ModelBase):
    info = ModelInfo(
        name="YoloV6N0_2_1",
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

    return non_maximum_suppression2(outputs, iou_thres=0.7)


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


class YoloV8S_decoding(ModelBase):
    info = ModelInfo(name="YoloV8S_decoding", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

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


class YoloV9_GELAN_C(ModelBase):
    info = ModelInfo(name="YoloV9_GELAN_C", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

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


class YoloV9M(ModelBase):
    info = ModelInfo(name="YoloV9M", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

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


def yolov11_postprocessing(outputs: List[np.ndarray]):
    outputs = outputs[0]
    outputs = torch.from_numpy(outputs)
    outputs = outputs.transpose(1, 2)

    return non_maximum_suppression2(outputs, iou_thres=0.7)


class YoloV11N(ModelBase):
    info = ModelInfo(name="YoloV11N", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

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


class YoloV11S(ModelBase):
    info = ModelInfo(name="YoloV11S", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

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


class YoloV11M(ModelBase):
    info = ModelInfo(name="YoloV11M", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

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


class YoloV11L(ModelBase):
    info = ModelInfo(name="YoloV11L", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

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


class YoloV11X(ModelBase):
    info = ModelInfo(name="YoloV11X", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

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


class YOLOV3_608_PPU(ModelBase):
    info = ModelInfo(
        name="YOLOV3_608_PPU",
        dataset=DatasetType.coco,
        evaluation=EvaluationType.coco,
        raw_performance="",
        q_lite_performance="",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)
        self.evaluator.lazy_postprocessing = True
        self.evaluator.use_ppu = True
        self.evaluator.anchors = [
            [(10, 13), (16, 30), (33, 23)],
            [(30, 61), (62, 45), (59, 119)],
            [(116, 90), (156, 198), (373, 326)],
        ]
        self.evaluator.yolo_version = "v3"

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=608, pad_location="edge", pad_value=[114, 114, 114]),
                Div(255),
                ConvertColor("BGR2RGB"),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=608, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return yolo_postprocessing


class YOLOV3_416_PPU(ModelBase):
    info = ModelInfo(
        name="YOLOV3_416_PPU",
        dataset=DatasetType.coco,
        evaluation=EvaluationType.coco,
        raw_performance="",
        q_lite_performance="",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)
        self.evaluator.lazy_postprocessing = True
        self.evaluator.use_ppu = True
        self.evaluator.anchors = [
            [(10, 13), (16, 30), (33, 23)],
            [(30, 61), (62, 45), (59, 119)],
            [(116, 90), (156, 198), (373, 326)],
        ]
        self.evaluator.yolo_version = "v3"

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


class YOLOV3Tiny_PPU(ModelBase):
    info = ModelInfo(
        name="YOLOV3Tiny_PPU",
        dataset=DatasetType.coco,
        evaluation=EvaluationType.coco,
        raw_performance="",
        q_lite_performance="",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)
        self.evaluator.use_padding = False
        self.evaluator.lazy_postprocessing = True
        self.evaluator.use_ppu = True
        self.evaluator.anchors = [
            [(10, 14), (23, 27), (37, 58)],
            [(81, 82), (135, 169), (344, 319)],
        ]
        self.evaluator.strides = [16, 32]
        self.evaluator.yolo_version = "v3"

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


class YOLOV4_PPU(ModelBase):
    info = ModelInfo(
        name="YOLOV4_PPU",
        dataset=DatasetType.coco,
        evaluation=EvaluationType.coco,
        raw_performance="",
        q_lite_performance="",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)
        self.evaluator.lazy_postprocessing = True
        self.evaluator.use_ppu = True
        self.evaluator.anchors = [
            [(12, 16), (19, 36), (40, 28)],
            [(36, 75), (76, 55), (72, 146)],
            [(142, 110), (192, 243), (459, 401)],
        ]
        self.evaluator.yolo_version = "v4"

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=512, pad_location="edge", pad_value=[114, 114, 114]),
                Div(255),
                ConvertColor("BGR2RGB"),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=512, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return yolo_postprocessing


class YOLOV5S_PPU(ModelBase):
    info = ModelInfo(
        name="YOLOV5S_PPU",
        dataset=DatasetType.coco,
        evaluation=EvaluationType.coco,
        raw_performance="",
        q_lite_performance="",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)
        self.evaluator.lazy_postprocessing = True
        self.evaluator.use_ppu = True
        self.evaluator.anchors = [
            [(10, 13), (16, 30), (33, 23)],
            [(30, 61), (62, 45), (59, 119)],
            [(116, 90), (156, 198), (373, 326)],
        ]
        self.evaluator.yolo_version = "v5"

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


class YOLOV7_PPU(ModelBase):
    info = ModelInfo(
        name="YOLOV7_PPU",
        dataset=DatasetType.coco,
        evaluation=EvaluationType.coco,
        raw_performance="",
        q_lite_performance="",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)
        self.evaluator.lazy_postprocessing = True
        self.evaluator.use_ppu = True
        self.evaluator.anchors = [
            [(12, 16), (19, 36), (40, 28)],
            [(36, 75), (76, 55), (72, 146)],
            [(142, 110), (192, 243), (459, 401)],
        ]
        self.evaluator.yolo_version = "v7"

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


class YOLOV7X_PPU(ModelBase):
    info = ModelInfo(
        name="YOLOV7X_PPU",
        dataset=DatasetType.coco,
        evaluation=EvaluationType.coco,
        raw_performance="",
        q_lite_performance="",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)
        self.evaluator.lazy_postprocessing = True
        self.evaluator.use_ppu = True
        self.evaluator.anchors = [
            [(12, 16), (19, 36), (40, 28)],
            [(36, 75), (76, 55), (72, 146)],
            [(142, 110), (192, 243), (459, 401)],
        ]
        self.evaluator.yolo_version = "v7"

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


class YOLOV8N_PPU(ModelBase):
    info = ModelInfo(
        name="YOLOV8N_PPU",
        dataset=DatasetType.coco,
        evaluation=EvaluationType.coco,
        raw_performance="",
        q_lite_performance="",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)
        self.evaluator.lazy_postprocessing = True
        self.evaluator.use_ppu = True
        self.evaluator.box_format = "cxcywh"

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


class YOLOV8S_PPU(ModelBase):
    info = ModelInfo(
        name="YOLOV8S_PPU",
        dataset=DatasetType.coco,
        evaluation=EvaluationType.coco,
        raw_performance="",
        q_lite_performance="",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)
        self.evaluator.lazy_postprocessing = True
        self.evaluator.use_ppu = True
        self.evaluator.box_format = "cxcywh"

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


class YOLOV9T_PPU(ModelBase):
    info = ModelInfo(
        name="YOLOV9T_PPU",
        dataset=DatasetType.coco,
        evaluation=EvaluationType.coco,
        raw_performance="",
        q_lite_performance="",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)
        self.evaluator.lazy_postprocessing = True
        self.evaluator.use_ppu = True
        self.evaluator.box_format = "cxcywh"

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


class YOLOXS_PPU(ModelBase):
    info = ModelInfo(
        name="YOLOXS_PPU",
        dataset=DatasetType.coco,
        evaluation=EvaluationType.coco,
        raw_performance="",
        q_lite_performance="",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)
        self.evaluator.lazy_postprocessing = True
        self.evaluator.use_ppu = True
        self.evaluator.yolo_version = "x"

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
            [
                Resize(mode="pad", size=640, pad_location="edge", pad_value=[114, 114, 114]),
                Transpose([2, 0, 1]),
            ]
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


class YOLOV11N_PPU(ModelBase):
    info = ModelInfo(
        name="YOLOV11N_PPU",
        dataset=DatasetType.coco,
        evaluation=EvaluationType.coco,
        raw_performance="",
        q_lite_performance="",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)
        self.evaluator.lazy_postprocessing = True
        self.evaluator.use_ppu = True
        self.evaluator.box_format = "cxcywh"

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


class YOLOV12N_PPU(ModelBase):
    info = ModelInfo(
        name="YOLOV12N_PPU",
        dataset=DatasetType.coco,
        evaluation=EvaluationType.coco,
        raw_performance="",
        q_lite_performance="",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)
        self.evaluator.lazy_postprocessing = True
        self.evaluator.use_ppu = True
        self.evaluator.box_format = "cxcywh"

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


# ==================== Additional YOLO Models ====================


def yolov3_gluon_postprocessing(anchors, strides, input_size):
    """YOLOv3 Gluon style postprocessing for multi-scale outputs."""

    def _postprocessing(outputs: List[np.ndarray]):
        all_predictions = []

        for scale_idx, output in enumerate(outputs):
            output = torch.from_numpy(output)
            # output shape: [B, H, W, 255] -> [B, H, W, 3, 85]
            B, H, W, C = output.shape
            num_anchors = 3
            num_classes = 80
            output = output.view(B, H, W, num_anchors, 5 + num_classes)

            # Create grid
            stride = strides[scale_idx]
            yv, xv = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
            grid = torch.stack((xv, yv), dim=-1).float()

            # Decode
            anchor = torch.tensor(anchors[scale_idx]).float()

            # xy = (sigmoid(xy) + grid) * stride
            xy = (torch.sigmoid(output[..., :2]) + grid.unsqueeze(2)) * stride
            # wh = exp(wh) * anchor
            wh = torch.exp(output[..., 2:4]) * anchor
            # objectness and class scores
            conf = torch.sigmoid(output[..., 4:5])
            cls_scores = torch.sigmoid(output[..., 5:])

            # Reshape to [B, -1, 85]
            predictions = torch.cat([xy, wh, conf, cls_scores], dim=-1)
            predictions = predictions.view(B, -1, 5 + num_classes)
            all_predictions.append(predictions)

        detections = torch.cat(all_predictions, dim=1)

        return non_maximum_suppression(detections)

    return _postprocessing


class YoloV3_Gluon_416(ModelBase):
    info = ModelInfo(name="YoloV3_Gluon_416", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

    def __init__(self, evaluator):
        # Anchors for YOLOv3 Gluon (scaled by stride)
        self.anchors = [
            [[116, 90], [156, 198], [373, 326]],  # stride 32
            [[30, 61], [62, 45], [59, 119]],  # stride 16
            [[10, 13], [16, 30], [33, 23]],  # stride 8
        ]
        self.strides = [32, 16, 8]
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=416, pad_location="edge", pad_value=[114, 114, 114]),
                Div(255),
                ConvertColor("BGR2RGB"),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=416, pad_location="edge", pad_value=[114, 114, 114]),
                Div(255),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return yolov3_gluon_postprocessing(self.anchors, self.strides, 416)


class YoloV3_Gluon_608(ModelBase):
    info = ModelInfo(name="YoloV3_Gluon_608", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

    def __init__(self, evaluator):
        self.anchors = [
            [[116, 90], [156, 198], [373, 326]],
            [[30, 61], [62, 45], [59, 119]],
            [[10, 13], [16, 30], [33, 23]],
        ]
        self.strides = [32, 16, 8]
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=608, pad_location="edge", pad_value=[114, 114, 114]),
                Div(255),
                ConvertColor("BGR2RGB"),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=608, pad_location="edge", pad_value=[114, 114, 114]),
                Div(255),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return yolov3_gluon_postprocessing(self.anchors, self.strides, 608)


def yolov4_postprocessing(input_size: int):
    """YOLOv4 postprocessing for boxes/confs separate outputs.

    Args:
        input_size: Input image size for scaling normalized coordinates to pixels
    """

    def _yolov4_postprocessing(outputs: List[np.ndarray]):
        # outputs[0]: boxes [B, N, 1, 4] in xyxy format (normalized 0~1)
        # outputs[1]: confs [B, N, 80] class scores
        boxes = torch.from_numpy(outputs[0]).squeeze(2)  # [B, N, 4]
        confs = torch.from_numpy(outputs[1])  # [B, N, 80]

        # Scale normalized coordinates to pixel coordinates
        boxes = boxes * input_size

        # Combine: [x1, y1, x2, y2, class_scores] for non_maximum_suppression2
        combined = torch.cat([boxes, confs], dim=-1)
        return non_maximum_suppression2(combined, cxcywh2xyxy_conversion=False)

    return _yolov4_postprocessing


class YoloV4_640(ModelBase):
    info = ModelInfo(name="YoloV4_640", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

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
        return yolov4_postprocessing(640)


class YoloV4_Leaky_512(ModelBase):
    info = ModelInfo(name="YoloV4_Leaky_512", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=512, pad_location="edge", pad_value=[114, 114, 114]),
                Div(255),
                ConvertColor("BGR2RGB"),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=512, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return yolov4_postprocessing(512)


class YoloV4Tiny_416(ModelBase):
    info = ModelInfo(name="YoloV4Tiny_416", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

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
        return yolov4_postprocessing(416)


class YoloV5L_640(ModelBase):
    info = ModelInfo(name="YoloV5L_640", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

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


class YoloV5M_640(ModelBase):
    info = ModelInfo(name="YoloV5M_640", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

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


class YoloV5S_640(ModelBase):
    info = ModelInfo(name="YoloV5S_640", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

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


class YoloV5X_640(ModelBase):
    info = ModelInfo(name="YoloV5X_640", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

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


class YoloV5L6_1280(ModelBase):
    info = ModelInfo(name="YoloV5L6_1280", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

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


class YoloV5M6_1(ModelBase):
    info = ModelInfo(name="YoloV5M6_1", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

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


class YoloV5M6_1280(ModelBase):
    info = ModelInfo(name="YoloV5M6_1280", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

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


class YoloV5M6_61_1280(ModelBase):
    info = ModelInfo(name="YoloV5M6_61_1280", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

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


class YoloV5N6_1280(ModelBase):
    info = ModelInfo(name="YoloV5N6_1280", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

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


class YoloV5N6_61_1280(ModelBase):
    info = ModelInfo(name="YoloV5N6_61_1280", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

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


class YoloV5S6_1280(ModelBase):
    info = ModelInfo(name="YoloV5S6_1280", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

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


class YoloV5S6_61_1280(ModelBase):
    info = ModelInfo(name="YoloV5S6_61_1280", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

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


class YoloV5X6_1280(ModelBase):
    info = ModelInfo(name="YoloV5X6_1280", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

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


def yolov5_multiscale_postprocessing(anchors, strides, input_size):
    """YOLOv5 multi-scale output postprocessing."""

    def _postprocessing(outputs: List[np.ndarray]):
        all_predictions = []

        for scale_idx, output in enumerate(outputs):
            output = torch.from_numpy(output)
            # output shape: [B, 3, H, W, 85]
            B, num_anchors, H, W, C = output.shape
            output = output.permute(0, 2, 3, 1, 4)  # [B, H, W, 3, 85]

            stride = strides[scale_idx]
            yv, xv = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
            grid = torch.stack((xv, yv), dim=-1).float()

            anchor = torch.tensor(anchors[scale_idx]).float()

            # Decode
            xy = (torch.sigmoid(output[..., :2]) * 2 - 0.5 + grid.unsqueeze(2)) * stride
            wh = (torch.sigmoid(output[..., 2:4]) * 2) ** 2 * anchor
            conf = torch.sigmoid(output[..., 4:5])
            cls_scores = torch.sigmoid(output[..., 5:])

            predictions = torch.cat([xy, wh, conf, cls_scores], dim=-1)
            predictions = predictions.view(B, -1, C)
            all_predictions.append(predictions)

        detections = torch.cat(all_predictions, dim=1)
        return non_maximum_suppression(detections)

    return _postprocessing


class YoloV5M_WoSpp_640(ModelBase):
    info = ModelInfo(name="YoloV5M_WoSpp_640", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

    def __init__(self, evaluator):
        self.anchors = [
            [[10, 13], [16, 30], [33, 23]],
            [[30, 61], [62, 45], [59, 119]],
            [[116, 90], [156, 198], [373, 326]],
        ]
        self.strides = [8, 16, 32]
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
        return yolov5_multiscale_postprocessing(self.anchors, self.strides, 640)


class YoloV5S_WoSpp_640(ModelBase):
    info = ModelInfo(name="YoloV5S_WoSpp_640", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

    def __init__(self, evaluator):
        self.anchors = [
            [[116, 90], [156, 198], [373, 326]],
            [[30, 61], [62, 45], [59, 119]],
            [[10, 13], [16, 30], [33, 23]],
        ]
        self.strides = [32, 16, 8]
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
        return yolov5_multiscale_postprocessing(self.anchors, self.strides, 640)


class YoloV5S_BboxDecoding_640(ModelBase):
    info = ModelInfo(name="YoloV5S_BboxDecoding_640", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

    def __init__(self, evaluator):
        self.anchors = [
            [[10, 13], [16, 30], [33, 23]],
            [[30, 61], [62, 45], [59, 119]],
            [[116, 90], [156, 198], [373, 326]],
        ]
        self.strides = [8, 16, 32]
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
        return yolov5_multiscale_postprocessing(self.anchors, self.strides, 640)


class YoloV5S_C3tr_640(ModelBase):
    info = ModelInfo(name="YoloV5S_C3tr_640", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

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


class YoloV5XS_WoSpp_512(ModelBase):
    info = ModelInfo(name="YoloV5XS_WoSpp_512", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

    def __init__(self, evaluator):
        self.anchors = [
            [[116, 90], [156, 198], [373, 326]],
            [[30, 61], [62, 45], [59, 119]],
            [[10, 13], [16, 30], [33, 23]],
        ]
        self.strides = [32, 16, 8]
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=512, pad_location="edge", pad_value=[114, 114, 114]),
                Div(255),
                ConvertColor("BGR2RGB"),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=512, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return yolov5_multiscale_postprocessing(self.anchors, self.strides, 512)


class YoloV6L_640(ModelBase):
    info = ModelInfo(name="YoloV6L_640", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

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


class YoloV6M_640(ModelBase):
    info = ModelInfo(name="YoloV6M_640", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

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


class YoloV6S_640(ModelBase):
    info = ModelInfo(name="YoloV6S_640", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

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


class YoloV6N6_1280(ModelBase):
    info = ModelInfo(name="YoloV6N6_1280", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

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


class YoloV6S6_1280(ModelBase):
    info = ModelInfo(name="YoloV6S6_1280", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

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


class YoloV6N_NmsCore_640(ModelBase):
    info = ModelInfo(name="YoloV6N_NmsCore_640", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

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


class YoloV7D6_1280(ModelBase):
    info = ModelInfo(name="YoloV7D6_1280", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

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


class YoloV7E6E_1280(ModelBase):
    info = ModelInfo(name="YoloV7E6E_1280", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

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


class YoloXL_640(ModelBase):
    info = ModelInfo(name="YoloXL_640", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

    def __init__(self, evaluator):
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
        super().__init__(evaluator)

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


class YoloXM_640(ModelBase):
    info = ModelInfo(name="YoloXM_640", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

    def __init__(self, evaluator):
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
        super().__init__(evaluator)

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


class YoloXN_416(ModelBase):
    info = ModelInfo(name="YoloXN_416", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

    def __init__(self, evaluator):
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
        super().__init__(evaluator)

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


class YoloXTiny_416(ModelBase):
    info = ModelInfo(name="YoloXTiny_416", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

    def __init__(self, evaluator):
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
        super().__init__(evaluator)

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


class YoloXX_640(ModelBase):
    info = ModelInfo(name="YoloXX_640", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

    def __init__(self, evaluator):
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
        super().__init__(evaluator)

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


def yolo_deepx_postprocessing(outputs: List[np.ndarray]):
    """YOLO_DEEPX postprocessing:
    outputs[0]=class_scores [1,8400,80],
    outputs[1]=boxes [1,8400,4] in xyxy pixel coords
    """
    class_scores = torch.from_numpy(outputs[0])  # [1, 8400, 80]
    boxes = torch.from_numpy(outputs[1])  # [1, 8400, 4] in xyxy format (pixel coordinates)

    # Combine: [x1, y1, x2, y2, class_scores] for non_maximum_suppression2
    combined = torch.cat([boxes, class_scores], dim=-1)
    return non_maximum_suppression2(combined, cxcywh2xyxy_conversion=False)


class YOLO_DeepX_640(ModelBase):
    info = ModelInfo(
        name="YOLO_DeepX_640",
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
        return yolo_deepx_postprocessing
