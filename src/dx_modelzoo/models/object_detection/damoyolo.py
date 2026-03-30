# src/dx_modelzoo/models/object_detection/damoyolo.py

import numpy as np
import torch
import torchvision
from torchvision.transforms import Compose

from dx_modelzoo.enums import DatasetType, EvaluationType
from dx_modelzoo.models import ModelBase, ModelInfo
from dx_modelzoo.preprocessing.convertcolor import ConvertColor
from dx_modelzoo.preprocessing.div import Div
from dx_modelzoo.preprocessing.enums import ResizeMode
from dx_modelzoo.preprocessing.resize import Resize
from dx_modelzoo.preprocessing.transpose import Transpose


def damoyolo_postprocessing(outputs, score_threshold=0.001, nms_threshold=0.7):
    """
    Postprocessing for DAMO-YOLO with exported outputs.
    Supports both single tensor (batch, 8400, 85) and split outputs.
    Applies unified NMS for better performance.
    """
    # num_classes = 80

    # Handle different output formats
    if len(outputs) == 1:
        # Single tensor: (batch, 8400, 85) or (8400, 85)
        pred = outputs[0]
        if isinstance(pred, np.ndarray):
            pred = torch.from_numpy(pred)

        # Remove batch dimension if present
        if pred.dim() == 3:
            pred = pred[0]  # (8400, 85)

        # Split: boxes (4) + objectness (1) + class scores (80)
        boxes = pred[:, :4]  # (8400, 4) - xyxy format
        obj_scores = pred[:, 4]  # (8400,) - objectness
        cls_scores = pred[:, 5:]  # (8400, 80) - class scores

        # Combine objectness and class scores
        scores = obj_scores.unsqueeze(1) * cls_scores  # (8400, 80)

    else:
        # Split outputs: separate boxes and scores
        b0, b1 = outputs[0], outputs[1]

        # Identify which is boxes vs scores by shape
        if isinstance(b0, np.ndarray):
            b0_shape = b0.shape
        else:
            b0_shape = tuple(b0.shape)

        if b0_shape[-1] == 4:
            boxes = b0
            scores = b1
        else:
            boxes = b1
            scores = b0

        # Remove batch dimension if present
        if isinstance(boxes, np.ndarray) and boxes.ndim == 3:
            boxes = boxes[0]
        if isinstance(scores, np.ndarray) and scores.ndim == 3:
            scores = scores[0]
        if isinstance(boxes, torch.Tensor) and boxes.dim() == 3:
            boxes = boxes[0]
        if isinstance(scores, torch.Tensor) and scores.dim() == 3:
            scores = scores[0]

        # Convert to tensors
        if isinstance(boxes, np.ndarray):
            boxes = torch.from_numpy(boxes).to(torch.float32)
        if isinstance(scores, np.ndarray):
            scores = torch.from_numpy(scores).to(torch.float32)

    # Ensure float32
    boxes = boxes.to(torch.float32)
    scores = scores.to(torch.float32)

    # Guard: check shapes
    if boxes.numel() == 0 or scores.numel() == 0:
        return torch.empty((0, 6))

    # Unified NMS approach (more efficient than per-class)
    # Get best class for each prediction
    best_scores, best_classes = scores.max(dim=1)  # (8400,)

    # Apply confidence threshold
    mask = best_scores > score_threshold
    if not mask.any():
        return torch.empty((0, 6))

    boxes = boxes[mask]
    best_scores = best_scores[mask]
    best_classes = best_classes[mask]

    # Apply NMS
    keep = torchvision.ops.nms(boxes, best_scores, nms_threshold)
    if keep.numel() == 0:
        return torch.empty((0, 6))

    # Format output: [x1, y1, x2, y2, confidence, class_id]
    final_boxes = boxes[keep]
    final_scores = best_scores[keep]
    final_classes = best_classes[keep].float()

    result = torch.cat([final_boxes, final_scores.unsqueeze(1), final_classes.unsqueeze(1)], dim=1)
    return result


class DamoYoloL(ModelBase):
    info = ModelInfo(
        name="DamoYoloL",
        dataset=DatasetType.coco,
        evaluation=EvaluationType.coco,
        raw_performance="",
        q_lite_performance="",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        """
        Preprocessing pipeline for ONNX Runtime. Includes letterbox resize,
        normalization, and transpose.
        """
        return Compose(
            [
                Resize(mode=ResizeMode.pad, size=640, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor("BGR2RGB"),
                Div(255),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        """
        Preprocessing pipeline for NPU. Letterbox resize and color conversion.
        Arithmetic operations are handled by the hardware.
        """
        return Compose(
            [
                Resize(mode=ResizeMode.pad, size=640, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        """
        Returns the DAMO-Yolo specific postprocessing function.
        """
        return damoyolo_postprocessing


class DamoYoloM(ModelBase):
    info = ModelInfo(
        name="DamoYoloM",
        dataset=DatasetType.coco,
        evaluation=EvaluationType.coco,
        raw_performance="",
        q_lite_performance="",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        """
        Preprocessing pipeline for ONNX Runtime. Includes letterbox resize,
        normalization, and transpose.
        """
        return Compose(
            [
                Resize(mode=ResizeMode.pad, size=640, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor("BGR2RGB"),
                Div(255),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        """
        Preprocessing pipeline for NPU. Letterbox resize and color conversion.
        Arithmetic operations are handled by the hardware.
        """
        return Compose(
            [
                Resize(mode=ResizeMode.pad, size=640, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        """
        Returns the DAMO-Yolo specific postprocessing function.
        """
        return damoyolo_postprocessing
    

class damoyolo_tinynas_l20_m(ModelBase):
    info = ModelInfo(
        name="damoyolo_tinynas_l20_m",
        dataset=DatasetType.coco,
        evaluation=EvaluationType.coco,
        raw_performance="",
        q_lite_performance="",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        """
        Preprocessing pipeline for ONNX Runtime. Includes letterbox resize,
        normalization, and transpose.
        """
        return Compose(
            [
                Resize(mode=ResizeMode.pad, size=640, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor("BGR2RGB"),
                Div(255),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        """
        Preprocessing pipeline for NPU. Letterbox resize and color conversion.
        Arithmetic operations are handled by the hardware.
        """
        return Compose(
            [
                Resize(mode=ResizeMode.pad, size=640, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        """
        Returns the DAMO-Yolo specific postprocessing function.
        """
        return damoyolo_postprocessing


class DamoYoloS(ModelBase):
    info = ModelInfo(
        name="DamoYoloS",
        dataset=DatasetType.coco,
        evaluation=EvaluationType.coco,
        raw_performance="",
        q_lite_performance="",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        """
        Preprocessing pipeline for ONNX Runtime. Includes letterbox resize,
        normalization, and transpose.
        """
        return Compose(
            [
                Resize(mode=ResizeMode.pad, size=640, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor("BGR2RGB"),
                Div(255),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        """
        Preprocessing pipeline for NPU. Letterbox resize and color conversion.
        Arithmetic operations are handled by the hardware.
        """
        return Compose(
            [
                Resize(mode=ResizeMode.pad, size=640, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        """
        Returns the DAMO-Yolo specific postprocessing function.
        """
        return damoyolo_postprocessing
    

class damoyolo_tinynas_l25_s(ModelBase):
    info = ModelInfo(
        name="damoyolo_tinynas_l25_s",
        dataset=DatasetType.coco,
        evaluation=EvaluationType.coco,
        raw_performance="",
        q_lite_performance="",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        """
        Preprocessing pipeline for ONNX Runtime. Includes letterbox resize,
        normalization, and transpose.
        """
        return Compose(
            [
                Resize(mode=ResizeMode.pad, size=640, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor("BGR2RGB"),
                Div(255),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        """
        Preprocessing pipeline for NPU. Letterbox resize and color conversion.
        Arithmetic operations are handled by the hardware.
        """
        return Compose(
            [
                Resize(mode=ResizeMode.pad, size=640, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        """
        Returns the DAMO-Yolo specific postprocessing function.
        """
        return damoyolo_postprocessing


class DamoYoloT(ModelBase):
    info = ModelInfo(
        name="DamoYoloT",
        dataset=DatasetType.coco,
        evaluation=EvaluationType.coco,
        raw_performance="",
        q_lite_performance="",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        """
        Preprocessing pipeline for ONNX Runtime. Includes letterbox resize,
        normalization, and transpose.
        """
        return Compose(
            [
                Resize(mode=ResizeMode.pad, size=640, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor("BGR2RGB"),
                Div(255),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        """
        Preprocessing pipeline for NPU. Letterbox resize and color conversion.
        Arithmetic operations are handled by the hardware.
        """
        return Compose(
            [
                Resize(mode=ResizeMode.pad, size=640, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        """
        Returns the DAMO-Yolo specific postprocessing function.
        """
        return damoyolo_postprocessing


class damoyolo_tinynas_l20_t(ModelBase):
    info = ModelInfo(
        name="damoyolo_tinynas_l20_t",
        dataset=DatasetType.coco,
        evaluation=EvaluationType.coco,
        raw_performance="",
        q_lite_performance="",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        """
        Preprocessing pipeline for ONNX Runtime. Includes letterbox resize,
        normalization, and transpose.
        """
        return Compose(
            [
                Resize(mode=ResizeMode.pad, size=640, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor("BGR2RGB"),
                Div(255),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        """
        Preprocessing pipeline for NPU. Letterbox resize and color conversion.
        Arithmetic operations are handled by the hardware.
        """
        return Compose(
            [
                Resize(mode=ResizeMode.pad, size=640, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        """
        Returns the DAMO-Yolo specific postprocessing function.
        """
        return damoyolo_postprocessing
