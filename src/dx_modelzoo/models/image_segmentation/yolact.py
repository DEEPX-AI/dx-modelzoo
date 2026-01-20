import numpy as np
import torch
from torchvision.transforms import Compose

from dx_modelzoo.enums import DatasetType, EvaluationType
from dx_modelzoo.evaluator import EvaluatorBase
from dx_modelzoo.models import ModelBase, ModelInfo
from dx_modelzoo.preprocessing.convertcolor import ConvertColor
from dx_modelzoo.preprocessing.div import Div
from dx_modelzoo.preprocessing.resize import Resize
from dx_modelzoo.preprocessing.transpose import Transpose


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _jaccard(box_a, box_b, iscrowd=False):
    """Compute IoU between two sets of boxes."""
    use_batch = True
    if len(box_a.shape) == 2:
        use_batch = False
        box_a = box_a[None, ...]
        box_b = box_b[None, ...]

    # Intersection
    max_xy = np.minimum(np.expand_dims(box_a[:, :, 2:], axis=2), np.expand_dims(box_b[:, :, 2:], axis=1))
    min_xy = np.maximum(np.expand_dims(box_a[:, :, :2], axis=2), np.expand_dims(box_b[:, :, :2], axis=1))
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=None)
    inter = inter[:, :, :, 0] * inter[:, :, :, 1]

    area_a = np.expand_dims((box_a[:, :, 2] - box_a[:, :, 0]) * (box_a[:, :, 3] - box_a[:, :, 1]), axis=2)
    area_b = np.expand_dims((box_b[:, :, 2] - box_b[:, :, 0]) * (box_b[:, :, 3] - box_b[:, :, 1]), axis=1)
    union = area_a + area_b - inter

    out = inter / area_a if iscrowd else inter / union

    return out if use_batch else np.squeeze(out, axis=0)


def yolact_postprocessing(predictions, conf_thresh=0.05, nms_thresh=0.5, top_k=200, max_num_detections=100):
    """
    YOLACT postprocessing for combined predictions tensor.

    Args:
        predictions: torch.Tensor of shape (batch, num_priors, 117)
                     where 117 = boxes(4) + class_scores(81) + mask_coeffs(32)

    Returns:
        List of torch.Tensor with shape (N, 4+1+1+32) = (N, 38)
        where each row is [x1, y1, x2, y2, score, class_id, mask_coeffs(32)]
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.numpy()

    batch_size = predictions.shape[0]

    # Split predictions: boxes(4) + class_scores(81) + mask_coeffs(32) = 117
    boxes = predictions[:, :, :4]  # (batch, 49104, 4)
    class_scores = predictions[:, :, 4:85]  # (batch, 49104, 81)
    mask_coeffs = predictions[:, :, 85:]  # (batch, 49104, 32)

    # num_classes = class_scores.shape[2]  # 81

    outputs = []

    for batch_idx in range(batch_size):
        batch_boxes = boxes[batch_idx]  # (49104, 4)
        batch_scores = class_scores[batch_idx]  # (49104, 81)
        batch_masks = mask_coeffs[batch_idx]  # (49104, 32)

        # Exclude background class (class 0), take max score across classes 1-80
        cur_scores = batch_scores[:, 1:]  # (49104, 80)
        conf_scores = np.amax(cur_scores, axis=1)  # (49104,)
        class_ids = np.argmax(cur_scores, axis=1)  # (49104,) - class index (0-79)

        # Filter by confidence threshold
        keep = conf_scores > conf_thresh
        if np.sum(keep) == 0:
            outputs.append(torch.zeros((0, 38)))
            continue

        filtered_boxes = batch_boxes[keep]  # (N, 4)
        filtered_scores = conf_scores[keep]  # (N,)
        filtered_classes = class_ids[keep]  # (N,)
        filtered_masks = batch_masks[keep]  # (N, 32)

        # Apply NMS per class
        result = _class_nms(
            filtered_boxes,
            filtered_masks,
            filtered_scores,
            filtered_classes,
            iou_threshold=nms_thresh,
            max_num_detections=max_num_detections,
        )

        if result is None:
            outputs.append(torch.zeros((0, 38)))
            continue

        det_boxes, det_masks, det_classes, det_scores = result

        # Format output: [x1, y1, x2, y2, score, class_id, mask_coeffs(32)]
        # class_id needs +1 to account for background class offset removed earlier
        output = np.concatenate(
            [
                det_boxes,  # (N, 4)
                det_scores[:, None],  # (N, 1)
                det_classes[:, None],  # (N, 1) - already 0-79, will be mapped in evaluator
                det_masks,  # (N, 32)
            ],
            axis=1,
        )

        outputs.append(torch.from_numpy(output.astype(np.float32)))

    return outputs


def _class_nms(boxes, masks, scores, classes, iou_threshold=0.5, max_num_detections=100):
    """
    Apply NMS across all detections.

    Args:
        boxes: (N, 4) - decoded boxes
        masks: (N, 32) - mask coefficients
        scores: (N,) - confidence scores
        classes: (N,) - class indices (0-79)
    """
    if len(scores) == 0:
        return None

    # Sort by score
    idx = np.argsort(scores)[::-1]
    boxes = boxes[idx]
    masks = masks[idx]
    scores = scores[idx]
    classes = classes[idx]

    # Simple NMS
    keep = []
    while len(boxes) > 0:
        keep.append(0)
        if len(boxes) == 1:
            break

        # Compute IoU with the rest
        ious = _compute_iou(boxes[0:1], boxes[1:])

        # Keep boxes with IoU <= threshold OR different class
        same_class = classes[1:] == classes[0]
        remove = (ious > iou_threshold) & same_class

        # Update arrays
        remain = ~remove
        boxes = boxes[1:][remain]
        masks = masks[1:][remain]
        scores = scores[1:][remain]
        classes = classes[1:][remain]

        if len(keep) >= max_num_detections:
            break

    # Gather kept detections
    idx = np.argsort(scores)[::-1]
    original_boxes = boxes
    original_masks = masks
    original_scores = scores
    original_classes = classes

    # Re-sort and select from original filtered data
    idx = np.argsort(scores)[::-1]
    boxes = boxes[idx]
    masks = masks[idx]
    scores = scores[idx]
    classes = classes[idx]

    # Actually we need to rebuild from keep indices
    # Let me fix this logic
    return _class_nms_v2(
        original_boxes, original_masks, original_scores, original_classes, iou_threshold, max_num_detections
    )


def _class_nms_v2(boxes, masks, scores, classes, iou_threshold=0.5, max_num_detections=100):
    """
    Apply NMS across all detections - corrected version.
    """
    if len(scores) == 0:
        return None

    # Sort by score descending
    order = np.argsort(scores)[::-1]
    boxes = boxes[order]
    masks = masks[order]
    scores = scores[order]
    classes = classes[order]

    keep_boxes = []
    keep_masks = []
    keep_scores = []
    keep_classes = []

    suppressed = np.zeros(len(boxes), dtype=bool)

    for i in range(len(boxes)):
        if suppressed[i]:
            continue

        keep_boxes.append(boxes[i])
        keep_masks.append(masks[i])
        keep_scores.append(scores[i])
        keep_classes.append(classes[i])

        if len(keep_boxes) >= max_num_detections:
            break

        # Suppress overlapping boxes of the same class
        for j in range(i + 1, len(boxes)):
            if suppressed[j]:
                continue
            if classes[j] != classes[i]:
                continue

            iou = _compute_iou_single(boxes[i], boxes[j])
            if iou > iou_threshold:
                suppressed[j] = True

    if len(keep_boxes) == 0:
        return None

    return (
        np.array(keep_boxes),
        np.array(keep_masks),
        np.array(keep_classes),
        np.array(keep_scores),
    )


def _compute_iou(box_a, boxes_b):
    """Compute IoU between one box and multiple boxes."""
    # box_a: (1, 4), boxes_b: (N, 4)
    x1 = np.maximum(box_a[0, 0], boxes_b[:, 0])
    y1 = np.maximum(box_a[0, 1], boxes_b[:, 1])
    x2 = np.minimum(box_a[0, 2], boxes_b[:, 2])
    y2 = np.minimum(box_a[0, 3], boxes_b[:, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    area_a = (box_a[0, 2] - box_a[0, 0]) * (box_a[0, 3] - box_a[0, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    union = area_a + area_b - inter
    return inter / (union + 1e-6)


def _compute_iou_single(box_a, box_b):
    """Compute IoU between two boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    union = area_a + area_b - inter
    return inter / (union + 1e-6)


class YOLACT_RegNetX_1_6gf(ModelBase):
    info = ModelInfo("YOLACT_RegNetX_1_6gf", dataset=DatasetType.coco, evaluation=EvaluationType.instance_segmentation)

    def __init__(self, evaluator: EvaluatorBase) -> None:
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                ConvertColor("BGR2RGB"),
                Resize(mode="default", size=512, interpolation="BILINEAR"),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                ConvertColor("BGR2RGB"),
                Resize(mode="pad", size=512, pad_location="edge", pad_value=[114, 114, 114]),
            ]
        )

    def postprocessing(self):
        return yolact_postprocessing


class YOLACT_RegNetX_800mf(ModelBase):
    info = ModelInfo("YOLACT_RegNetX_800mf", dataset=DatasetType.coco, evaluation=EvaluationType.instance_segmentation)

    def __init__(self, evaluator: EvaluatorBase) -> None:
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                ConvertColor("BGR2RGB"),
                Resize(mode="pad", size=512, pad_location="edge", pad_value=[114, 114, 114]),
                Div(255),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                ConvertColor("BGR2RGB"),
                Resize(mode="pad", size=512, pad_location="edge", pad_value=[114, 114, 114]),
            ]
        )

    def postprocessing(self):
        return yolact_postprocessing
