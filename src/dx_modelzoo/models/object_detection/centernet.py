"""CenterNet object detection models.

CenterNet uses heatmap-based detection with center points and size regression.
"""
from typing import List

import numpy as np
import torch
from torchvision.ops import nms
from torchvision.transforms import Compose

from dx_modelzoo.enums import DatasetType, EvaluationType
from dx_modelzoo.models import ModelBase, ModelInfo
from dx_modelzoo.preprocessing.convertcolor import ConvertColor
from dx_modelzoo.preprocessing.div import Div
from dx_modelzoo.preprocessing.resize import Resize

MAX_NMS = 3000


def centernet_postprocessing(
    outputs: List[np.ndarray],
    input_size: int = 512,
    num_classes: int = 80,
    score_threshold: float = 0.1,
    nms_threshold: float = 0.5,
    max_detections: int = 300,
) -> torch.Tensor:
    """
    CenterNet postprocessing.

    CenterNet outputs:
    - offset (wh): [1, H, W, 2] - width/height offsets
    - regression (reg): [1, H, W, 2] - center point offsets
    - heatmap: [1, H, W, num_classes] - class heatmaps (after ReLU, so 0~1 confidence)

    Args:
        outputs: List of numpy arrays [offset, regression, heatmap]
        input_size: Input image size (512)
        num_classes: Number of classes (80 for COCO)
        score_threshold: Minimum confidence threshold
        nms_threshold: NMS IoU threshold
        max_detections: Maximum number of detections

    Returns:
        Tensor of shape (N, 6) with [x1, y1, x2, y2, score, class_id]
    """
    # CenterNet outputs: [offset(wh), reg, heatmap]
    # offset: [1, 128, 128, 2] - width, height
    # reg: [1, 128, 128, 2] - center offset x, y
    # heatmap: [1, 128, 128, 80] - class confidence

    if len(outputs) != 3:
        return torch.empty((0, 6))

    # Identify outputs by shape
    offset = None
    reg = None
    heatmap = None

    for out in outputs:
        if out.shape[-1] == num_classes:
            heatmap = out
        elif out.shape[-1] == 2:
            if offset is None:
                offset = out
            else:
                reg = out

    if heatmap is None or offset is None or reg is None:
        return torch.empty((0, 6))

    # Squeeze batch dimension
    heatmap = heatmap[0]  # [H, W, C]
    offset = offset[0]  # [H, W, 2] - width, height
    reg = reg[0]  # [H, W, 2] - center offset x, y

    H, W, C = heatmap.shape
    stride = input_size // H  # 512 / 128 = 4

    # Find local maxima in heatmap (simple peak detection)
    # For efficiency, we'll use a simpler approach: find all points above threshold

    # Get max class score and class index for each spatial location
    max_scores = np.max(heatmap, axis=-1)  # [H, W]
    class_indices = np.argmax(heatmap, axis=-1)  # [H, W]

    # Find points above threshold
    mask = max_scores > score_threshold
    if not np.any(mask):
        return torch.empty((0, 6))

    # Get coordinates of detected centers
    ys, xs = np.where(mask)
    scores = max_scores[mask]
    classes = class_indices[mask]

    # Get width/height and center offsets for detected points
    wh = offset[ys, xs]  # [N, 2] - width, height
    center_offset = reg[ys, xs]  # [N, 2] - offset x, y

    # Compute center coordinates in input image space
    # CenterNet uses (x + offset_x) * stride for center
    cx = (xs.astype(np.float32) + center_offset[:, 0]) * stride
    cy = (ys.astype(np.float32) + center_offset[:, 1]) * stride

    # Compute box coordinates
    # Width/height are in input image space
    w = wh[:, 0] * stride
    h = wh[:, 1] * stride

    x1 = np.clip(cx - w / 2, 0, input_size)
    y1 = np.clip(cy - h / 2, 0, input_size)
    x2 = np.clip(cx + w / 2, 0, input_size)
    y2 = np.clip(cy + h / 2, 0, input_size)

    # Convert to tensors
    boxes = torch.from_numpy(np.stack([x1, y1, x2, y2], axis=1)).float()
    scores_t = torch.from_numpy(scores).float()
    classes_t = torch.from_numpy(classes).long()

    if boxes.shape[0] == 0:
        return torch.empty((0, 6))

    num_boxes = boxes.size(0)
    if num_boxes > MAX_NMS:
        topk_indices = scores_t.topk(MAX_NMS).indices
        boxes = boxes[topk_indices]
        scores_t = scores_t[topk_indices]
        classes_t = classes_t[topk_indices]

    # Apply class-aware NMS
    class_offset = classes_t.float().unsqueeze(1) * (input_size * 2)
    boxes_for_nms = boxes + class_offset

    keep = nms(boxes_for_nms, scores_t, nms_threshold)
    keep = keep[:max_detections]

    # Combine results
    result = torch.cat(
        [
            boxes[keep],
            scores_t[keep].unsqueeze(1),
            classes_t[keep].unsqueeze(1).float(),
        ],
        dim=1,
    )

    return result


class CenterNet_ResNet18(ModelBase):
    """CenterNet with ResNet18 backbone."""

    info = ModelInfo(
        name="CenterNet_ResNet18",
        dataset=DatasetType.coco,
        evaluation=EvaluationType.coco,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        # Input: NHWC format [1, 512, 512, 3], RGB, 0-255
        return Compose(
            [
                Resize(mode="pad", size=512, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor("BGR2RGB"),
                Div(255),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=512, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor("BGR2RGB"),
                Div(255),
            ]
        )

    def postprocessing(self):
        return lambda outputs: centernet_postprocessing(
            outputs,
            input_size=512,
            num_classes=80,
            score_threshold=0.001,
            nms_threshold=0.7,
        )


class CenterNet_ResNet50(ModelBase):
    """CenterNet with ResNet50 backbone."""

    info = ModelInfo(
        name="CenterNet_ResNet50",
        dataset=DatasetType.coco,
        evaluation=EvaluationType.coco,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        # Input: NHWC format [1, 512, 512, 3], RGB, 0-255
        return Compose(
            [
                Resize(mode="pad", size=512, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor("BGR2RGB"),
                Div(255),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=512, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor("BGR2RGB"),
                Div(255),
            ]
        )

    def postprocessing(self):
        return lambda outputs: centernet_postprocessing(
            outputs,
            input_size=512,
            num_classes=80,
            score_threshold=0.001,
            nms_threshold=0.7,
        )
