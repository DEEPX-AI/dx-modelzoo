from typing import List

import numpy as np
import torch
from torchvision.transforms import Compose

from dx_modelzoo.enums import DatasetType, EvaluationType
from dx_modelzoo.models import ModelBase, ModelInfo
from dx_modelzoo.models.object_detection.nms import non_maximum_suppression, non_maximum_suppression2
from dx_modelzoo.preprocessing.convertcolor import ConvertColor
from dx_modelzoo.preprocessing.div import Div
from dx_modelzoo.preprocessing.resize import Resize
from dx_modelzoo.preprocessing.transpose import Transpose


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
        return yolo_postprocessing


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


class YoloV10(ModelBase):
    info = ModelInfo(name="YoloV10", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

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
