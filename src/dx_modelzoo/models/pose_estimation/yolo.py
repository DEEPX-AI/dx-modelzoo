from typing import List

import numpy as np
import torch
from torchvision.transforms import Compose

from dx_modelzoo.enums import DatasetType, EvaluationType
from dx_modelzoo.models import ModelBase, ModelInfo
from dx_modelzoo.models.object_detection.nms import non_maximum_suppression_for_pose
from dx_modelzoo.preprocessing.convertcolor import ConvertColor
from dx_modelzoo.preprocessing.div import Div
from dx_modelzoo.preprocessing.resize import Resize
from dx_modelzoo.preprocessing.transpose import Transpose


def make_yolov8_pose_postprocessing(conf_threshold: float, iou_threshold: float):
    def _postproc(outputs: List[np.ndarray]):
        raw = outputs[0]
        preds = torch.from_numpy(raw) if not isinstance(raw, torch.Tensor) else raw

        # Normalize to [B, N, 56]
        if preds.dim() == 2:
            if preds.shape[0] == 56:
                preds = preds.transpose(0, 1).unsqueeze(0)
            else:
                preds = preds.unsqueeze(0)
        elif preds.dim() == 3:
            if preds.shape[1] == 56:
                preds = preds.transpose(1, 2)

        kept = non_maximum_suppression_for_pose(
            preds,
            conf_thres=conf_threshold,
            iou_thres=iou_threshold,
            num_classes=0,  # Pose models don't have class probabilities
        )
        if kept.numel() == 0:
            return (
                torch.empty((0, 4)),
                torch.empty(0),
                torch.empty(0),
                torch.empty((0, 17, 3)),
            )

        boxes_xyxy = kept[:, :4]
        scores = kept[:, 4]
        class_ids = kept[:, 5]
        extras = kept[:, 6:]
        keypoints = extras.reshape(-1, 17, 3).clone()

        return boxes_xyxy, scores, class_ids, keypoints

    return _postproc


class YOLOV8N_Pose(ModelBase):
    info = ModelInfo(name="YOLOV8N_Pose", dataset=DatasetType.coco_pose, evaluation=EvaluationType.coco_pose)

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
        # Use evaluator thresholds if available
        conf_t = getattr(
            self.evaluator,
            "conf_threshold",
            0.25,
        )
        iou_t = getattr(self.evaluator, "iou_threshold", 0.45)
        return make_yolov8_pose_postprocessing(conf_t, iou_t)


class YOLOV8S_Pose(ModelBase):
    info = ModelInfo(name="YOLOV8S_Pose", dataset=DatasetType.coco_pose, evaluation=EvaluationType.coco_pose)

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
        conf_t = getattr(self.evaluator, "conf_threshold", 0.25)
        iou_t = getattr(self.evaluator, "iou_threshold", 0.7)
        return make_yolov8_pose_postprocessing(conf_t, iou_t)


class YOLOV8M_Pose(ModelBase):
    info = ModelInfo(name="YOLOV8M_Pose", dataset=DatasetType.coco_pose, evaluation=EvaluationType.coco_pose)

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
        conf_t = getattr(self.evaluator, "conf_threshold", 0.25)
        iou_t = getattr(self.evaluator, "iou_threshold", 0.7)
        return make_yolov8_pose_postprocessing(conf_t, iou_t)


class YOLOV8L_Pose(ModelBase):
    info = ModelInfo(name="YOLOV8L_Pose", dataset=DatasetType.coco_pose, evaluation=EvaluationType.coco_pose)

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
        conf_t = getattr(self.evaluator, "conf_threshold", 0.25)
        iou_t = getattr(self.evaluator, "iou_threshold", 0.7)
        return make_yolov8_pose_postprocessing(conf_t, iou_t)


class YOLOV8X_Pose(ModelBase):
    info = ModelInfo(name="YOLOV8X_Pose", dataset=DatasetType.coco_pose, evaluation=EvaluationType.coco_pose)

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
        conf_t = getattr(self.evaluator, "conf_threshold", 0.25)
        iou_t = getattr(self.evaluator, "iou_threshold", 0.7)
        return make_yolov8_pose_postprocessing(conf_t, iou_t)


def yolov11_pose_postprocessing(outputs: List[np.ndarray], conf_thres: float = 0.30, iou_thres: float = 0.50):
    """YOLOv11 Pose postprocessing.

    Args:
        outputs: Model outputs
        conf_thres: Confidence threshold
        iou_thres: IoU threshold for NMS

    Returns:
        Detections with keypoints: [x1, y1, x2, y2, score, class, kpt_x1, kpt_y1, ..., kpt_x8, kpt_y8]
    """
    if not isinstance(outputs, list):
        outputs = [outputs]

    outputs = outputs[0]  # (1, C, N) or (C, N)
    if outputs.ndim == 3:
        outputs = outputs[0]  # (C, N)

    outputs = torch.from_numpy(outputs) if isinstance(outputs, np.ndarray) else outputs

    if outputs.shape[0] < outputs.shape[1]:
        outputs = outputs.T

    # Format: [cx, cy, w, h, cls0, cls1, ..., clsN, kpt1_x, kpt1_y, kpt1_v, ...]
    N, C = outputs.shape

    # For 4 classes and 8 keypoints: 4 + 4 + 8*3 = 32
    bbox = outputs[:, :4]  # [cx, cy, w, h]

    remaining = C - 4
    if remaining >= 4 + 8 * 3:  # 4 classes, 8 keypoints
        num_classes = 4
        num_keypoints = 8
    elif remaining >= 1 + 17 * 3:  # 1 class, 17 keypoints (COCO pose)
        num_classes = 1
        num_keypoints = 17
    else:
        num_classes = min(80, remaining // 4)
        remaining_after_cls = remaining - num_classes
        num_keypoints = remaining_after_cls // 3

    class_scores = outputs[:, 4 : 4 + num_classes]
    keypoints = outputs[:, 4 + num_classes :].reshape(N, num_keypoints, 3)  # [N, K, 3]

    scores, class_ids = torch.max(class_scores, dim=1)

    mask = scores > conf_thres
    if mask.sum() == 0:
        return torch.empty((0, 6 + num_keypoints * 2), dtype=torch.float32)

    bbox = bbox[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]
    keypoints = keypoints[mask]

    # Convert bbox from [cx, cy, w, h] to [x1, y1, x2, y2]
    x1 = bbox[:, 0] - bbox[:, 2] / 2
    y1 = bbox[:, 1] - bbox[:, 3] / 2
    x2 = bbox[:, 0] + bbox[:, 2] / 2
    y2 = bbox[:, 1] + bbox[:, 3] / 2
    boxes = torch.stack([x1, y1, x2, y2], dim=1)

    keep_indices = torch.ops.torchvision.nms(boxes, scores, iou_thres)

    boxes = boxes[keep_indices]
    scores = scores[keep_indices]
    class_ids = class_ids[keep_indices]
    keypoints = keypoints[keep_indices]

    # Flatten keypoints [N, K, 3] -> [N, K*2] (only x, y, ignore visibility)
    kpts_xy = keypoints[:, :, :2].reshape(keypoints.shape[0], -1)  # [N, K*2]

    # Concatenate: [x1, y1, x2, y2, score, class, kpt_x1, kpt_y1, ..., kpt_xK, kpt_yK]
    result = torch.cat([boxes, scores.unsqueeze(1), class_ids.unsqueeze(1).float(), kpts_xy], dim=1)

    return result


class YoloV11N_Pose(ModelBase):
    """YOLOv11 Pose model for pose estimation.

    Supports both 4-class custom poses and standard COCO 17-keypoint pose.
    Output format: [x1, y1, x2, y2, score, class, kpt_x1, kpt_y1, ..., kpt_xK, kpt_yK]
    """

    info = ModelInfo(name="YoloV11N_Pose", dataset=DatasetType.coco_pose, evaluation=EvaluationType.coco_pose)

    def __init__(self, evaluator, input_size=(288, 512), conf_thres=0.30, iou_thres=0.50):
        self.input_size = input_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="default", size=self.input_size),
                Div(255),
                ConvertColor("BGR2RGB"),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="default", size=self.input_size),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        conf_thres = self.conf_thres
        iou_thres = self.iou_thres

        def _post(outputs: List[np.ndarray]):
            result = yolov11_pose_postprocessing(outputs, conf_thres=conf_thres, iou_thres=iou_thres)

            # Return empty arrays if no detections
            if result.shape[0] == 0:
                return (
                    torch.empty((0, 4), dtype=torch.float32),  # boxes
                    torch.empty((0,), dtype=torch.float32),  # scores
                    torch.empty((0,), dtype=torch.long),  # class_ids
                    torch.empty((0, 0), dtype=torch.float32),  # keypoints
                )

            # Parse result: [x1, y1, x2, y2, score, class, kpt_x1, kpt_y1, ..., kpt_xK, kpt_yK]
            boxes = result[:, :4]  # [N, 4]
            scores = result[:, 4]  # [N]
            class_ids = result[:, 5].long()  # [N]
            keypoints = result[:, 6:]  # [N, K*2]

            return boxes, scores, class_ids, keypoints

        return _post


def yolo26_postprocessing(outputs: List[np.ndarray], conf_thres: float = 0.30):
    """
    COCO Pose postprocessing for (1, 300, 57) format output (NMS-free model).

    Format: [x, y, w, h, conf, class, kpt1_x, kpt1_y, kpt1_v, ..., kpt17_x, kpt17_y, kpt17_v]
    - Box: 4 (x, y, w, h in cxcywh format)
    - Confidence: 1
    - Class: 1
    - Keypoints: 17 × 3 = 51 (x, y, visibility for each keypoint)
    Total: 4 + 1 + 1 + 51 = 57

    Args:
        outputs: Model outputs [(1, 300, 57)]
        conf_thres: Confidence threshold

    Returns:
        Tuple of (boxes_xyxy, scores, class_ids, keypoints)
    """
    raw = outputs[0]  # (1, 300, 57)
    preds = torch.from_numpy(raw) if not isinstance(raw, torch.Tensor) else raw

    # Remove batch dimension if present
    if preds.dim() == 3:
        preds = preds.squeeze(0)  # (300, 57)

    # Filter by confidence
    scores = preds[:, 4]  # (300,)
    mask = scores > conf_thres

    if mask.sum() == 0:
        return (
            torch.empty((0, 4), dtype=torch.float32),  # boxes
            torch.empty((0,), dtype=torch.float32),  # scores
            torch.empty((0,), dtype=torch.long),  # class_ids
            torch.empty((0, 17, 3), dtype=torch.float32),  # keypoints
        )

    filtered_preds = preds[mask]  # (N, 57)

    # Parse components
    boxes_cxcywh = filtered_preds[:, :4]  # (N, 4) [cx, cy, w, h]
    scores_filtered = filtered_preds[:, 4]  # (N,)
    class_ids = filtered_preds[:, 5].long()  # (N,)
    keypoints_flat = filtered_preds[:, 6:]  # (N, 51)

    # Reshape keypoints to (N, 17, 3)
    keypoints = keypoints_flat.reshape(-1, 17, 3)  # (N, 17, 3)

    # Convert boxes from cxcywh to xyxy
    cx, cy, w, h = boxes_cxcywh.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)  # (N, 4)

    # NMS-free model: return all filtered detections
    return boxes_xyxy, scores_filtered, class_ids, keypoints


class Yolo26N_Pose(ModelBase):
    info = ModelInfo(name="Yolo26N_Pose", dataset=DatasetType.coco_pose, evaluation=EvaluationType.coco_pose)

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
        return yolo26_postprocessing


class Yolo26S_Pose(ModelBase):
    info = ModelInfo(name="Yolo26S_Pose", dataset=DatasetType.coco_pose, evaluation=EvaluationType.coco_pose)

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
        return yolo26_postprocessing


class Yolo26M_Pose(ModelBase):
    info = ModelInfo(name="Yolo26M_Pose", dataset=DatasetType.coco_pose, evaluation=EvaluationType.coco_pose)

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
        return yolo26_postprocessing


class Yolo26L_Pose(ModelBase):
    info = ModelInfo(name="Yolo26L_Pose", dataset=DatasetType.coco_pose, evaluation=EvaluationType.coco_pose)

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
        return yolo26_postprocessing


class Yolo26X_Pose(ModelBase):
    info = ModelInfo(name="Yolo26X_Pose", dataset=DatasetType.coco_pose, evaluation=EvaluationType.coco_pose)

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
        return yolo26_postprocessing
