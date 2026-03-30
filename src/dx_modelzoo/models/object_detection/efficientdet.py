import numpy as np
import torch
from torchvision.ops import nms
from torchvision.transforms import Compose

from dx_modelzoo.enums import DatasetType, EvaluationType
from dx_modelzoo.models import ModelBase, ModelInfo
from dx_modelzoo.preprocessing.convertcolor import ConvertColor
from dx_modelzoo.preprocessing.div import Div
from dx_modelzoo.preprocessing.resize import Resize
from dx_modelzoo.preprocessing.transpose import Transpose


def generate_anchors(input_size=640):
    levels = [3, 4, 5, 6, 7]
    anchor_scale = 4.0
    scales = [2**0, 2 ** (1 / 3), 2 ** (2 / 3)]
    aspect_ratios = [1.0, 2.0, 0.5]

    all_anchors = []

    for level in levels:
        stride = 2**level
        grid_size = input_size // stride

        shifts = (torch.arange(0, grid_size) + 0.5) * stride
        shift_y, shift_x = torch.meshgrid(shifts, shifts, indexing="ij")
        centers = torch.stack([shift_y, shift_x, shift_y, shift_x], dim=-1).reshape(-1, 4)

        anchor_deltas = []
        for scale in scales:
            for ratio in aspect_ratios:
                w_ratio = np.sqrt(ratio)
                h_ratio = 1.0 / w_ratio

                side = stride * anchor_scale * scale
                hh = side * h_ratio
                ww = side * w_ratio

                anchor_deltas.append(torch.tensor([-hh / 2, -ww / 2, hh / 2, ww / 2]))

        anchor_deltas = torch.stack(anchor_deltas).to(torch.float32)

        level_anchors = centers.unsqueeze(1) + anchor_deltas.unsqueeze(0)
        all_anchors.append(level_anchors.reshape(-1, 4))

    anchors = torch.cat(all_anchors, dim=0).unsqueeze(0).numpy().astype(np.float32)
    return anchors


def d1_postprocessing(outputs, input_size=640, score_th=0.001, iou_th=0.7, topk_pre=2000, final_max=300):
    """
    EfficientDet postprocessing with optimized box decoding and class-aware NMS.

    Args:
        outputs: Model outputs where last 3 tensors are [regression, classification, anchors]
        input_size: Input image size for clipping boxes
        score_th: Score threshold for filtering detections
        iou_th: IoU threshold for NMS
        topk_pre: Maximum number of boxes to keep before NMS
        final_max: Maximum number of final detections

    Returns:
        Tensor of shape (N, 6) with [x1, y1, x2, y2, score, class_id]
    """
    if outputs[-3].shape[-1] == 4:
        regression, classification = outputs[-3:-1]
    else:
        regression, classification = outputs[-2:]
    anchors = generate_anchors(input_size=input_size)

    reg_t = torch.from_numpy(regression).squeeze(0)  # (N, 4)
    cls_t = torch.from_numpy(classification).squeeze(0)  # (N, C)
    anc_t = torch.from_numpy(anchors).squeeze(0)  # (N, 4)

    # Decode bounding boxes from anchors and regression
    # Anchors format: [y1, x1, y2, x2]
    # Regression format: [dy, dx, dh, dw]
    wa = anc_t[:, 3] - anc_t[:, 1]  # width
    ha = anc_t[:, 2] - anc_t[:, 0]  # height
    a_cx = (anc_t[:, 1] + anc_t[:, 3]) * 0.5  # center x
    a_cy = (anc_t[:, 0] + anc_t[:, 2]) * 0.5  # center y

    pred_cx = reg_t[:, 1] * wa + a_cx
    pred_cy = reg_t[:, 0] * ha + a_cy
    pred_w = torch.exp(reg_t[:, 3]) * wa
    pred_h = torch.exp(reg_t[:, 2]) * ha

    # Convert to [x1, y1, x2, y2] format
    x1 = (pred_cx - 0.5 * pred_w).clamp(0, input_size)
    y1 = (pred_cy - 0.5 * pred_h).clamp(0, input_size)
    x2 = (pred_cx + 0.5 * pred_w).clamp(0, input_size)
    y2 = (pred_cy + 0.5 * pred_h).clamp(0, input_size)

    boxes = torch.stack([x1, y1, x2, y2], dim=1)  # (N, 4)

    # Apply sigmoid to classification scores (EfficientDet uses sigmoid)
    scores = cls_t

    # Get max score and class index for each box
    max_scores, class_indices = scores.max(dim=1)  # (N,), (N,)

    # Filter by score threshold
    score_mask = max_scores > score_th
    boxes = boxes[score_mask]
    max_scores = max_scores[score_mask]
    class_indices = class_indices[score_mask]

    if boxes.shape[0] == 0:
        return torch.empty((0, 6))

    # Keep top-k before NMS
    if boxes.shape[0] > topk_pre:
        topk_scores, topk_idx = max_scores.topk(topk_pre)
        boxes = boxes[topk_idx]
        max_scores = topk_scores
        class_indices = class_indices[topk_idx]

    # Class-aware NMS: offset boxes by class ID
    class_offset = class_indices.float().unsqueeze(1) * (input_size * 2)
    boxes_for_nms = boxes + class_offset

    # Apply NMS
    keep = nms(boxes_for_nms, max_scores, iou_th)

    # Limit to final_max detections
    keep = keep[:final_max]

    # Combine results: [x1, y1, x2, y2, score, class_id]
    # Note: EfficientDet uses 90 COCO classes (1-90), class_indices is 0-indexed, so add 1
    final = torch.cat(
        [
            boxes[keep],
            max_scores[keep].unsqueeze(1),
            (class_indices[keep] + 1).unsqueeze(1).float(),  # Convert 0-indexed to 1-indexed (COCO format)
        ],
        dim=1,
    )

    return final


class EfficientDet_D1(ModelBase):
    info = ModelInfo(
        name="EfficientDet_D1",
        dataset=DatasetType.coco,
        evaluation=EvaluationType.coco,
        raw_performance="",
        q_lite_performance="",
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)
        self.evaluator.use_class_90 = True

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
        return lambda x: d1_postprocessing(x, input_size=640)


def d0_postprocessing(outputs, input_size=512, score_th=0.001, iou_th=0.7, max_output=300):
    """
    EfficientDet-D0 postprocessing with anchor decoding.

    D0 outputs:
    - regression: [1, 49104, 4] - offset from anchors [dy, dx, dh, dw]
    - confidence: [1, 49104, 1] - objectness scores
    """
    if len(outputs) < 2:
        return torch.empty((0, 6))

    # Find regression and confidence by shape
    reg_out = None
    conf_out = None

    for out in outputs:
        if out.shape[-1] == 4:
            reg_out = out
        elif out.shape[-1] == 1:
            conf_out = out

    if reg_out is None or conf_out is None:
        return torch.empty((0, 6))

    reg_t = torch.from_numpy(reg_out).squeeze(0)  # [N, 4]
    conf_t = torch.from_numpy(conf_out).squeeze(0).squeeze(-1)  # [N]

    # Generate anchors
    anchors = generate_anchors(input_size=input_size)
    anc_t = torch.from_numpy(anchors).squeeze(0)  # [N, 4]

    # Decode boxes from anchors and regression
    # Anchors format: [y1, x1, y2, x2]
    wa = anc_t[:, 3] - anc_t[:, 1]  # width
    ha = anc_t[:, 2] - anc_t[:, 0]  # height
    a_cx = (anc_t[:, 1] + anc_t[:, 3]) * 0.5  # center x
    a_cy = (anc_t[:, 0] + anc_t[:, 2]) * 0.5  # center y

    # Regression format: [dy, dx, dh, dw]
    pred_cx = reg_t[:, 1] * wa + a_cx
    pred_cy = reg_t[:, 0] * ha + a_cy
    pred_w = torch.exp(reg_t[:, 3].clamp(-10, 10)) * wa
    pred_h = torch.exp(reg_t[:, 2].clamp(-10, 10)) * ha

    # Convert to [x1, y1, x2, y2]
    x1 = (pred_cx - 0.5 * pred_w).clamp(0, input_size)
    y1 = (pred_cy - 0.5 * pred_h).clamp(0, input_size)
    x2 = (pred_cx + 0.5 * pred_w).clamp(0, input_size)
    y2 = (pred_cy + 0.5 * pred_h).clamp(0, input_size)

    boxes = torch.stack([x1, y1, x2, y2], dim=1)  # [N, 4]

    # Filter by score threshold
    score_mask = conf_t > score_th
    boxes = boxes[score_mask]
    scores = conf_t[score_mask]

    if boxes.shape[0] == 0:
        return torch.empty((0, 6))

    # Apply NMS
    keep = nms(boxes, scores, iou_th)
    keep = keep[:max_output]

    # D0 doesn't output class info, so use class 1 (person) as default
    result = torch.cat(
        [
            boxes[keep],
            scores[keep].unsqueeze(1),
            torch.ones(len(keep), 1),  # class 1
        ],
        dim=1,
    )

    return result


class EfficientDet_D0(ModelBase):
    info = ModelInfo(
        name="EfficientDet_D0",
        dataset=DatasetType.coco,
        evaluation=EvaluationType.coco,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)
        self.evaluator.use_class_90 = True

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=512, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor("BGR2RGB"),
                Div(255),
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
        return lambda x: d0_postprocessing(x, input_size=512)


class EfficientDet_D2(ModelBase):
    info = ModelInfo(
        name="EfficientDet_D2",
        dataset=DatasetType.coco,
        evaluation=EvaluationType.coco,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)
        self.evaluator.use_class_90 = True

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=768, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor("BGR2RGB"),
                Div(255),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=768, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return lambda x: d1_postprocessing(x, input_size=768)


class EfficientDet_D3(ModelBase):
    info = ModelInfo(
        name="EfficientDet_D3",
        dataset=DatasetType.coco,
        evaluation=EvaluationType.coco,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)
        self.evaluator.use_class_90 = True

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=896, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor("BGR2RGB"),
                Div(255),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=896, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return lambda x: d1_postprocessing(x, input_size=896)


class EfficientDet_D4(ModelBase):
    info = ModelInfo(
        name="EfficientDet_D4",
        dataset=DatasetType.coco,
        evaluation=EvaluationType.coco,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)
        self.evaluator.use_class_90 = True

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=1024, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor("BGR2RGB"),
                Div(255),
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
        return lambda x: d1_postprocessing(x, input_size=1024)


class EfficientDet_D5(ModelBase):
    info = ModelInfo(
        name="EfficientDet_D5",
        dataset=DatasetType.coco,
        evaluation=EvaluationType.coco,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)
        self.evaluator.use_class_90 = True

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
        return lambda x: d1_postprocessing(x, input_size=1280)


class EfficientDet_D6(ModelBase):
    info = ModelInfo(
        name="EfficientDet_D6",
        dataset=DatasetType.coco,
        evaluation=EvaluationType.coco,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)
        self.evaluator.use_class_90 = True

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
        return lambda x: d1_postprocessing(x, input_size=1280)
