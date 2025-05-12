from typing import List

import numpy as np
import torch

from dx_modelzoo.enums import DatasetType, EvaluationType
from dx_modelzoo.models import ModelBase, ModelInfo
from dx_modelzoo.models.object_dection.nms import non_maximum_suppression, non_maximum_suppression2


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
        return [
            {
                "resize": {
                    "mode": "pad",
                    "size": 640,
                    "pad_location": "edge",
                    "pad_value": [114, 114, 114],
                }
            },
            {"div": {"x": 255}},
            {"convertColor": {"form": "BGR2RGB"}},
            {"transpose": {"axis": [2, 0, 1]}},
            {"expandDim": {"axis": 0}},
        ]

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
        return [
            {
                "resize": {
                    "mode": "pad",
                    "size": 640,
                    "pad_location": "edge",
                    "pad_value": [114, 114, 114],
                }
            },
            {"div": {"x": 255}},
            {"convertColor": {"form": "BGR2RGB"}},
            {"transpose": {"axis": [2, 0, 1]}},
            {"expandDim": {"axis": 0}},
        ]

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
        return [
            {
                "resize": {
                    "mode": "pad",
                    "size": 640,
                    "pad_location": "edge",
                    "pad_value": [114, 114, 114],
                }
            },
            {"div": {"x": 255}},
            {"convertColor": {"form": "BGR2RGB"}},
            {"transpose": {"axis": [2, 0, 1]}},
            {"expandDim": {"axis": 0}},
        ]

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
        return [
            {
                "resize": {
                    "mode": "pad",
                    "size": 640,
                    "pad_location": "edge",
                    "pad_value": [114, 114, 114],
                }
            },
            {"div": {"x": 255}},
            {"convertColor": {"form": "BGR2RGB"}},
            {"transpose": {"axis": [2, 0, 1]}},
            {"expandDim": {"axis": 0}},
        ]

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
        return [
            {
                "resize": {
                    "mode": "pad",
                    "size": 640,
                    "pad_location": "edge",
                    "pad_value": [114, 114, 114],
                }
            },
            {"div": {"x": 255}},
            {"convertColor": {"form": "BGR2RGB"}},
            {"transpose": {"axis": [2, 0, 1]}},
            {"expandDim": {"axis": 0}},
        ]

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
        return [
            {
                "resize": {
                    "mode": "pad",
                    "size": 640,
                    "pad_location": "edge",
                    "pad_value": [114, 114, 114],
                }
            },
            {"div": {"x": 255}},
            {"convertColor": {"form": "BGR2RGB"}},
            {"transpose": {"axis": [2, 0, 1]}},
            {"expandDim": {"axis": 0}},
        ]

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
        return [
            {
                "resize": {
                    "mode": "pad",
                    "size": 1280,
                    "pad_location": "edge",
                    "pad_value": [114, 114, 114],
                }
            },
            {"div": {"x": 255}},
            {"convertColor": {"form": "BGR2RGB"}},
            {"transpose": {"axis": [2, 0, 1]}},
            {"expandDim": {"axis": 0}},
        ]

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
        return [
            {
                "resize": {
                    "mode": "pad",
                    "size": 640,
                    "pad_location": "edge",
                    "pad_value": [114, 114, 114],
                }
            },
            {"div": {"x": 255}},
            {"convertColor": {"form": "BGR2RGB"}},
            {"transpose": {"axis": [2, 0, 1]}},
            {"expandDim": {"axis": 0}},
        ]

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
        return [
            {
                "resize": {
                    "mode": "pad",
                    "size": 640,
                    "pad_location": "edge",
                    "pad_value": [114, 114, 114],
                }
            },
            {"transpose": {"axis": [2, 0, 1]}},
            {"expandDim": {"axis": 0}},
        ]

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
        return [
            {"resize": {"mode": "pad", "size": 640, "pad_location": "edge", "pad_value": [114, 114, 114]}},
            {"div": {"x": 255}},
            {"convertColor": {"form": "BGR2RGB"}},
            {"transpose": {"axis": [2, 0, 1]}},
            {"expandDim": {"axis": 0}},
        ]

    def postprocessing(self):
        return yolov8_postprocessing


class YoloV8N(ModelBase):
    info = ModelInfo(name="YoloV8N", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return [
            {"resize": {"mode": "pad", "size": 640, "pad_location": "edge", "pad_value": [114, 114, 114]}},
            {"div": {"x": 255}},
            {"convertColor": {"form": "BGR2RGB"}},
            {"transpose": {"axis": [2, 0, 1]}},
            {"expandDim": {"axis": 0}},
        ]

    def postprocessing(self):
        return yolov8_postprocessing


class YoloV8S(ModelBase):
    info = ModelInfo(name="YoloV8S", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return [
            {"resize": {"mode": "pad", "size": 640, "pad_location": "edge", "pad_value": [114, 114, 114]}},
            {"div": {"x": 255}},
            {"convertColor": {"form": "BGR2RGB"}},
            {"transpose": {"axis": [2, 0, 1]}},
            {"expandDim": {"axis": 0}},
        ]

    def postprocessing(self):
        return yolov8_postprocessing


class YoloV8M(ModelBase):
    info = ModelInfo(name="YoloV8M", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return [
            {"resize": {"mode": "pad", "size": 640, "pad_location": "edge", "pad_value": [114, 114, 114]}},
            {"div": {"x": 255}},
            {"convertColor": {"form": "BGR2RGB"}},
            {"transpose": {"axis": [2, 0, 1]}},
            {"expandDim": {"axis": 0}},
        ]

    def postprocessing(self):
        return yolov8_postprocessing


class YoloV8L(ModelBase):
    info = ModelInfo(name="YoloV8L", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return [
            {
                "resize": {
                    "mode": "pad",
                    "size": 640,
                    "pad_location": "edge",
                    "pad_value": [114, 114, 114],
                }
            },
            {"div": {"x": 255}},
            {"convertColor": {"form": "BGR2RGB"}},
            {"transpose": {"axis": [2, 0, 1]}},
            {"expandDim": {"axis": 0}},
        ]

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
        return [
            {
                "resize": {
                    "mode": "pad",
                    "size": 640,
                    "pad_location": "edge",
                    "pad_value": [114, 114, 114],
                }
            },
            {"div": {"x": 255}},
            {"convertColor": {"form": "BGR2RGB"}},
            {"transpose": {"axis": [2, 0, 1]}},
            {"expandDim": {"axis": 0}},
        ]

    def postprocessing(self):
        return yolov9_postprocessing


class YoloV9S(ModelBase):
    info = ModelInfo(name="YoloV9S", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return [
            {
                "resize": {
                    "mode": "pad",
                    "size": 640,
                    "pad_location": "edge",
                    "pad_value": [114, 114, 114],
                }
            },
            {"div": {"x": 255}},
            {"convertColor": {"form": "BGR2RGB"}},
            {"transpose": {"axis": [2, 0, 1]}},
            {"expandDim": {"axis": 0}},
        ]

    def postprocessing(self):
        return yolov9_postprocessing


class YoloV9C(ModelBase):
    info = ModelInfo(name="YoloV9C", dataset=DatasetType.coco, evaluation=EvaluationType.coco)

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return [
            {
                "resize": {
                    "mode": "pad",
                    "size": 640,
                    "pad_location": "edge",
                    "pad_value": [114, 114, 114],
                }
            },
            {"div": {"x": 255}},
            {"convertColor": {"form": "BGR2RGB"}},
            {"transpose": {"axis": [2, 0, 1]}},
            {"expandDim": {"axis": 0}},
        ]

    def postprocessing(self):
        return yolov9_postprocessing
