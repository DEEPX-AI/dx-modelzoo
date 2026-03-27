import time
from typing import List, Literal, Tuple

import numpy as np
import torch
import torchvision

from dx_modelzoo.enums import SessionType
from dx_modelzoo.utils.detection import calculate_iou, convert_cxcywh_2_xyxy

MAX_WH = 4096  # 4k image size limit
MAX_NMS = 3000  # 8400  # 80^2 + 40^2 + 20^2


def get_confidence_scores(output: torch.Tensor) -> torch.Tensor:
    """caculate confidence score
    confidence scores is object confidence socres * class confidence scores.
    Args:
        output (torch.Tensor): confidence score.

    Returns:
        torch.Tensor: confidenc scores.
    """
    class_conf_scores = output[..., 4, None]
    obj_conf_scores = output[..., 5:]
    return obj_conf_scores * class_conf_scores


def filter_outputs_by_conf_score(
    outputs: torch.Tensor, confidence_scores: torch.Tensor, conf_thres: float, multi_label: bool
) -> torch.Tensor:
    """filter model outputs by confidence scores.

    Args:
        outputs (torch.Tensor): model outputs.
        confidence_scores (torch.Tensor): confidence scores.
        conf_thres (float): confidence score threshold.
        multi_label (bool): if True, consider all case of class.

    Returns:
        torch.Tensor: filtered outputs.
    """
    if multi_label:
        box_index, class_index = torch.where(confidence_scores > conf_thres)
        filtred_ouptuts = torch.cat(
            (
                outputs[box_index, :4],
                confidence_scores[box_index, class_index, None],
                class_index[:, None].float(),
            ),
            dim=1,
        )
    else:
        conf, class_index = confidence_scores.max(1, keepdim=True)
        filtred_ouptuts = torch.cat((outputs[..., :4], conf, class_index.float()), 1)[conf.view(-1) > conf_thres]
    return filtred_ouptuts


def non_maximum_suppression(
    outputs: List[np.ndarray | torch.Tensor],
    conf_thres: float = 0.001,
    iou_thres: float = 0.7,
    max_output_boxes: int = 300,
    multi_label: bool = False,
    cxcywh2xyxy_conversion: bool = True,
) -> torch.Tensor:
    """Perform Non-Maximum Suppression (NMS) to filter out redundant bounding boxes based on confidence scores and
    Intersection over Union (IoU).

    This function processes the model's output, which is in the center-width-height (cxcywh) format,
    and applies NMS to remove duplicate boxes that represent the same object in the image,
    keeping only the ones with the highest confidence scores.

    Args:
        outputs (List[np.ndarray]): A list containing the model's output predictions,
            where each element represents the predicted boxes in the format [cx, cy, w, h, conf, class].
        conf_thres (float, optional): Confidence score threshold. Boxes with a confidence score
            lower than this threshold are discarded. Default is 0.001.
        iou_thres (float, optional): Intersection over Union (IoU) threshold. Boxes with an IoU greater
            than this value are considered redundant and are suppressed. Default is 0.6.
        max_output_boxes (int, optional): The maximum number of boxes to return after NMS. Default is 300.

    Returns:
        torch.Tensor: A tensor containing the final set of boxes after applying NMS.
            The output has shape [num_boxes, 6] where each row represents a box,
            with the format [x_min, y_min, x_max, y_max, confidence_score, class_index].

    Notes:
        - The input boxes should be in the cxcywh format, and the output will be converted to xyxy format
          (i.e., [x_min, y_min, x_max, y_max]) before applying NMS.
        - The function assumes that `outputs` contains only one batch of predictions.
        - The class index is adjusted using a scaling factor `MAX_WH` to ensure that different classes
          do not interfere with the NMS process.
        - The function also uses `torchvision.ops.nms` to perform the NMS filtering.
    """
    # Convert to tensor if needed
    if not isinstance(outputs, torch.Tensor):
        outputs = torch.from_numpy(outputs[0])

    # Pre-compute confidence scores: obj_conf * class_conf (single vectorized operation)
    classes_score = outputs[..., 4:5] * outputs[..., 5:]
    max_scores, max_classes = classes_score.max(2)

    # Apply confidence threshold once
    mask = max_scores > conf_thres

    # Early exit if no detections
    if not mask.any():
        return torch.empty((0, 6), device=outputs.device)

    nms_outputs = []
    for batch in range(outputs.size(0)):
        batch_mask = mask[batch]
        if not batch_mask.any():
            continue

        # Extract only filtered data (minimize memory access)
        boxes = outputs[batch, batch_mask, :4]
        scores = max_scores[batch, batch_mask]
        class_indices = max_classes[batch, batch_mask]

        # Handle multi-label case
        if multi_label:
            batch_scores = classes_score[batch, batch_mask]
            box_indices, cls_indices = torch.where(batch_scores > conf_thres)
            boxes = boxes[box_indices]
            scores = batch_scores[box_indices, cls_indices]
            class_indices = cls_indices

        # Early exit if no boxes
        if boxes.size(0) == 0:
            continue

        # Convert coordinates in-place if needed
        if cxcywh2xyxy_conversion:
            boxes = convert_cxcywh_2_xyxy(boxes)

        # Limit boxes before NMS
        num_boxes = boxes.size(0)
        if num_boxes > MAX_NMS:
            topk_indices = scores.topk(MAX_NMS).indices
            boxes = boxes[topk_indices]
            scores = scores[topk_indices]
            class_indices = class_indices[topk_indices]

        # Use TorchNMS.batched_nms for better performance
        nms_indices = TorchNMS.batched_nms(boxes, scores, class_indices, iou_thres)

        # Limit final output
        if nms_indices.size(0) > max_output_boxes:
            nms_indices = nms_indices[:max_output_boxes]

        # Build output tensor with minimal operations
        selected_boxes = boxes[nms_indices]
        selected_scores = scores[nms_indices]
        selected_classes = class_indices[nms_indices].float()

        processed_output = torch.cat(
            [selected_boxes, selected_scores.unsqueeze(1), selected_classes.unsqueeze(1)], dim=1
        )

        nms_outputs.append(processed_output)

    # Return empty tensor if no detections
    if not nms_outputs:
        return torch.empty((0, 6), device=outputs.device)

    return torch.cat(nms_outputs, dim=0)


def non_maximum_suppression2(
    outputs: List[np.ndarray | torch.Tensor],
    conf_thres: float = 0.001,
    iou_thres: float = 0.7,
    max_output_boxes: int = 300,
    cxcywh2xyxy_conversion: bool = True,
) -> torch.Tensor:
    if not isinstance(outputs, torch.Tensor):
        outputs = torch.from_numpy(outputs[0])

    classes_score = outputs[..., 4:]
    max_scores, max_classes = classes_score.max(2)
    mask = max_scores > conf_thres

    nms_outputs = []
    for batch, output in enumerate(outputs):
        batch_mask = mask[batch]
        if not batch_mask.any():
            continue

        boxes = output[batch_mask, :4]
        scores = max_scores[batch, batch_mask]
        class_indices = max_classes[batch, batch_mask]

        if cxcywh2xyxy_conversion:
            boxes = convert_cxcywh_2_xyxy(boxes)

        num_boxes = boxes.size(0)
        if num_boxes > MAX_NMS:
            topk_scores, topk_indices = scores.topk(MAX_NMS)
            boxes = boxes[topk_indices]
            scores = topk_scores
            class_indices = class_indices[topk_indices]

        nms_output_index = torchvision.ops.nms(boxes + (class_indices.float().unsqueeze(1) * MAX_WH), scores, iou_thres)

        if nms_output_index.size(0) > max_output_boxes:
            nms_output_index = nms_output_index[:max_output_boxes]

        processed_output = torch.stack(
            [
                boxes[nms_output_index, 0],
                boxes[nms_output_index, 1],
                boxes[nms_output_index, 2],
                boxes[nms_output_index, 3],
                scores[nms_output_index],
                class_indices[nms_output_index].float(),
            ],
            dim=1,
        )

        nms_outputs.append(processed_output)

    if not nms_outputs:
        return torch.empty((0, 6))
    return torch.cat(nms_outputs, dim=0)


def calculate_ciou(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Calculate Complete IoU (CIoU) between two sets of boxes.

    Args:
        boxes1: (N, 4) tensor of boxes in xyxy format
        boxes2: (M, 4) tensor of boxes in xyxy format
        eps: small value to avoid division by zero

    Returns:
        (N, M) tensor of CIoU values
    """
    # Calculate IoU first
    x1_min, y1_min, x1_max, y1_max = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    x2_min, y2_min, x2_max, y2_max = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]

    # Intersection area
    inter_x1 = torch.max(x1_min.unsqueeze(1), x2_min.unsqueeze(0))
    inter_y1 = torch.max(y1_min.unsqueeze(1), y2_min.unsqueeze(0))
    inter_x2 = torch.min(x1_max.unsqueeze(1), x2_max.unsqueeze(0))
    inter_y2 = torch.min(y1_max.unsqueeze(1), y2_max.unsqueeze(0))

    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    # Union area
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1.unsqueeze(1) + area2.unsqueeze(0) - inter_area

    iou = inter_area / (union_area + eps)

    # Center distance
    c_x1 = (x1_min + x1_max) / 2
    c_y1 = (y1_min + y1_max) / 2
    c_x2 = (x2_min + x2_max) / 2
    c_y2 = (y2_min + y2_max) / 2

    center_dist = (c_x1.unsqueeze(1) - c_x2.unsqueeze(0)) ** 2 + (c_y1.unsqueeze(1) - c_y2.unsqueeze(0)) ** 2

    # Diagonal distance of the smallest enclosing box
    enclose_x1 = torch.min(x1_min.unsqueeze(1), x2_min.unsqueeze(0))
    enclose_y1 = torch.min(y1_min.unsqueeze(1), y2_min.unsqueeze(0))
    enclose_x2 = torch.max(x1_max.unsqueeze(1), x2_max.unsqueeze(0))
    enclose_y2 = torch.max(y1_max.unsqueeze(1), y2_max.unsqueeze(0))

    diag_dist = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2

    # Aspect ratio consistency
    w1 = x1_max - x1_min
    h1 = y1_max - y1_min
    w2 = x2_max - x2_min
    h2 = y2_max - y2_min

    v = (4 / (torch.pi**2)) * torch.pow(
        torch.atan(w2.unsqueeze(0) / (h2.unsqueeze(0) + eps)) - torch.atan(w1.unsqueeze(1) / (h1.unsqueeze(1) + eps)), 2
    )

    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    # CIoU
    ciou = iou - (center_dist / (diag_dist + eps) + alpha * v)

    return ciou


def calculate_diou(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Calculate Distance IoU (DIoU) between two sets of boxes.

    Args:
        boxes1: (N, 4) tensor of boxes in xyxy format
        boxes2: (M, 4) tensor of boxes in xyxy format
        eps: small value to avoid division by zero

    Returns:
        (N, M) tensor of DIoU values
    """
    # Calculate IoU first
    x1_min, y1_min, x1_max, y1_max = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    x2_min, y2_min, x2_max, y2_max = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]

    # Intersection area
    inter_x1 = torch.max(x1_min.unsqueeze(1), x2_min.unsqueeze(0))
    inter_y1 = torch.max(y1_min.unsqueeze(1), y2_min.unsqueeze(0))
    inter_x2 = torch.min(x1_max.unsqueeze(1), x2_max.unsqueeze(0))
    inter_y2 = torch.min(y1_max.unsqueeze(1), y2_max.unsqueeze(0))

    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    # Union area
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1.unsqueeze(1) + area2.unsqueeze(0) - inter_area

    iou = inter_area / (union_area + eps)

    # Center distance
    c_x1 = (x1_min + x1_max) / 2
    c_y1 = (y1_min + y1_max) / 2
    c_x2 = (x2_min + x2_max) / 2
    c_y2 = (y2_min + y2_max) / 2

    center_dist = (c_x1.unsqueeze(1) - c_x2.unsqueeze(0)) ** 2 + (c_y1.unsqueeze(1) - c_y2.unsqueeze(0)) ** 2

    # Diagonal distance of the smallest enclosing box
    enclose_x1 = torch.min(x1_min.unsqueeze(1), x2_min.unsqueeze(0))
    enclose_y1 = torch.min(y1_min.unsqueeze(1), y2_min.unsqueeze(0))
    enclose_x2 = torch.max(x1_max.unsqueeze(1), x2_max.unsqueeze(0))
    enclose_y2 = torch.max(y1_max.unsqueeze(1), y2_max.unsqueeze(0))

    diag_dist = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2

    # DIoU
    diou = iou - center_dist / (diag_dist + eps)

    return diou


def calculate_giou(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Calculate Generalized IoU (GIoU) between two sets of boxes.

    Args:
        boxes1: (N, 4) tensor of boxes in xyxy format
        boxes2: (M, 4) tensor of boxes in xyxy format
        eps: small value to avoid division by zero

    Returns:
        (N, M) tensor of GIoU values
    """
    # Calculate IoU first
    x1_min, y1_min, x1_max, y1_max = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    x2_min, y2_min, x2_max, y2_max = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]

    # Intersection area
    inter_x1 = torch.max(x1_min.unsqueeze(1), x2_min.unsqueeze(0))
    inter_y1 = torch.max(y1_min.unsqueeze(1), y2_min.unsqueeze(0))
    inter_x2 = torch.min(x1_max.unsqueeze(1), x2_max.unsqueeze(0))
    inter_y2 = torch.min(y1_max.unsqueeze(1), y2_max.unsqueeze(0))

    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    # Union area
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1.unsqueeze(1) + area2.unsqueeze(0) - inter_area

    iou = inter_area / (union_area + eps)

    # Smallest enclosing box area
    enclose_x1 = torch.min(x1_min.unsqueeze(1), x2_min.unsqueeze(0))
    enclose_y1 = torch.min(y1_min.unsqueeze(1), y2_min.unsqueeze(0))
    enclose_x2 = torch.max(x1_max.unsqueeze(1), x2_max.unsqueeze(0))
    enclose_y2 = torch.max(y1_max.unsqueeze(1), y2_max.unsqueeze(0))

    enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)

    # GIoU
    giou = iou - (enclose_area - union_area) / (enclose_area + eps)

    return giou


def batched_nms_with_iou_type(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_thres: float,
    iou_type: Literal["iou", "giou", "diou", "ciou"] = "iou",
) -> torch.Tensor:
    """
    Perform NMS with different IoU calculation methods.

    Args:
        boxes: (N, 4) tensor of boxes in xyxy format
        scores: (N,) tensor of scores
        iou_thres: IoU threshold for suppression
        iou_type: Type of IoU calculation ("iou", "giou", "diou", "ciou")

    Returns:
        Indices of boxes to keep
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    if iou_type == "iou":
        # Use torchvision's optimized NMS
        return torchvision.ops.nms(boxes, scores, iou_thres)

    # For other IoU types, implement custom NMS
    _, order = scores.sort(descending=True)
    keep = []

    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order[0].item())
            break

        i = order[0].item()
        keep.append(i)

        # Calculate IoU between current box and remaining boxes
        current_box = boxes[i : i + 1]
        other_boxes = boxes[order[1:]]

        if iou_type == "ciou":
            iou_values = calculate_ciou(current_box, other_boxes).squeeze(0)
        elif iou_type == "diou":
            iou_values = calculate_diou(current_box, other_boxes).squeeze(0)
        elif iou_type == "giou":
            iou_values = calculate_giou(current_box, other_boxes).squeeze(0)
        else:
            raise ValueError(f"Unknown iou_type: {iou_type}")

        # Keep boxes with IoU less than threshold
        mask = iou_values <= iou_thres
        order = order[1:][mask]

    return torch.tensor(keep, dtype=torch.int64, device=boxes.device)


def non_maximum_suppression_iou(
    outputs: List[np.ndarray | torch.Tensor],
    conf_thres: float = 0.001,
    iou_thres: float = 0.7,
    max_output_boxes: int = 300,
    multi_label: bool = False,
    cxcywh2xyxy_conversion: bool = True,
    iou_type: Literal["iou", "giou", "diou", "ciou"] = "iou",
) -> torch.Tensor:
    # Get thresholds from environment variables if set

    if not isinstance(outputs, torch.Tensor):
        outputs = torch.from_numpy(outputs[0])
    mask = outputs[..., 4] > conf_thres
    nms_outputs = []
    for batch, output in enumerate(outputs):
        output = output[mask[batch]]
        confidence_scores = get_confidence_scores(output)
        filtered_output = filter_outputs_by_conf_score(output, confidence_scores, conf_thres, multi_label)
        if cxcywh2xyxy_conversion:
            boxes = convert_cxcywh_2_xyxy(filtered_output[..., :4])
        else:
            boxes = filtered_output[..., :4]
        scores = filtered_output[..., 4, None]
        class_indices = filtered_output[..., 5, None]

        num_boxes = boxes.size(0)
        sorted_mask = scores[..., 0].argsort(descending=True)
        if num_boxes > MAX_NMS:
            sorted_mask = sorted_mask[:MAX_NMS]

        boxes = boxes[sorted_mask]
        scores = scores[sorted_mask]
        class_indices = class_indices[sorted_mask]

        # Use batched_nms_with_iou_type for different IoU calculations
        nms_output_index = batched_nms_with_iou_type(
            boxes + (class_indices * MAX_WH), scores[..., 0], iou_thres, iou_type=iou_type
        )

        num_nms_outputs = nms_output_index.size(0)
        if num_nms_outputs > max_output_boxes:
            nms_output_index = nms_output_index[:max_output_boxes]

        processed_output = torch.cat((boxes, scores, class_indices), dim=1)
        nms_outputs.append(processed_output[nms_output_index])
    return torch.concat(nms_outputs, axis=0)


def xywh2xyxy(boxes):
    """Convert boxes from [center_x, center_y, width, height] to [x1, y1, x2, y2] format."""
    if isinstance(boxes, torch.Tensor):
        result = boxes.clone()
    else:
        result = np.copy(boxes)

    half_w = boxes[..., 2] / 2
    half_h = boxes[..., 3] / 2
    result[..., 0] = boxes[..., 0] - half_w
    result[..., 1] = boxes[..., 1] - half_h
    result[..., 2] = boxes[..., 0] + half_w
    result[..., 3] = boxes[..., 1] + half_h
    return result


def box_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Calculate intersection-over-union (IoU) of boxes.

    Args:
        box1 (torch.Tensor): A tensor of shape (N, 4) representing N bounding boxes in (x1, y1, x2, y2) format.
        box2 (torch.Tensor): A tensor of shape (M, 4) representing M bounding boxes in (x1, y1, x2, y2) format.
        eps (float, optional): A small value to avoid division by zero.

    Returns:
        (torch.Tensor): An NxM tensor containing the pairwise IoU values for every element in box1 and box2.

    References:
        https://github.com/pytorch/vision/blob/main/torchvision/ops/boxes.py
    """
    # NOTE: Need .float() to get accurate iou values
    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.float().unsqueeze(1).chunk(2, 2), box2.float().unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


class TorchNMS:
    """NMS implementation optimized for YOLO.

    This class provides static methods for performing non-maximum suppression (NMS) operations on bounding boxes,
    including both standard NMS and batched NMS for multi-class scenarios.

    Methods:
        nms: Optimized NMS with early termination that matches torchvision behavior exactly.
        batched_nms: Batched NMS for class-aware suppression.

    Examples:
        Perform standard NMS on boxes and scores
        >>> boxes = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]])
        >>> scores = torch.tensor([0.9, 0.8])
        >>> keep = TorchNMS.nms(boxes, scores, 0.5)
    """

    @staticmethod
    def fast_nms(
        boxes: torch.Tensor,
        scores: torch.Tensor,
        iou_threshold: float,
        use_triu: bool = True,
        iou_func=box_iou,
        exit_early: bool = True,
    ) -> torch.Tensor:
        """Fast-NMS implementation from https://arxiv.org/pdf/1904.02689 using upper triangular matrix operations.

        Args:
            boxes (torch.Tensor): Bounding boxes with shape (N, 4) in xyxy format.
            scores (torch.Tensor): Confidence scores with shape (N,).
            iou_threshold (float): IoU threshold for suppression.
            use_triu (bool): Whether to use torch.triu operator for upper triangular matrix operations.
            iou_func (callable): Function to compute IoU between boxes.
            exit_early (bool): Whether to exit early if there are no boxes.

        Returns:
            (torch.Tensor): Indices of boxes to keep after NMS.

        Examples:
            Apply NMS to a set of boxes
            >>> boxes = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]])
            >>> scores = torch.tensor([0.9, 0.8])
            >>> keep = TorchNMS.nms(boxes, scores, 0.5)
        """
        if boxes.numel() == 0 and exit_early:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)

        sorted_idx = torch.argsort(scores, descending=True)
        boxes = boxes[sorted_idx]
        ious = iou_func(boxes, boxes)
        if use_triu:
            ious = ious.triu_(diagonal=1)
            # NOTE: handle the case when len(boxes) hence exportable by eliminating if-else condition
            pick = torch.nonzero((ious >= iou_threshold).sum(0) <= 0).squeeze_(-1)
        else:
            n = boxes.shape[0]
            row_idx = torch.arange(n, device=boxes.device).view(-1, 1).expand(-1, n)
            col_idx = torch.arange(n, device=boxes.device).view(1, -1).expand(n, -1)
            upper_mask = row_idx < col_idx
            ious = ious * upper_mask
            # Zeroing these scores ensures the additional indices would not affect the final results
            scores_ = scores[sorted_idx]
            scores_[~((ious >= iou_threshold).sum(0) <= 0)] = 0
            scores[sorted_idx] = scores_  # update original tensor for NMSModel
            # NOTE: return indices with fixed length to avoid TFLite reshape error
            pick = torch.topk(scores_, scores_.shape[0]).indices
        return sorted_idx[pick]

    @staticmethod
    def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
        """Optimized NMS with early termination that matches torchvision behavior exactly.

        Args:
            boxes (torch.Tensor): Bounding boxes with shape (N, 4) in xyxy format.
            scores (torch.Tensor): Confidence scores with shape (N,).
            iou_threshold (float): IoU threshold for suppression.

        Returns:
            (torch.Tensor): Indices of boxes to keep after NMS.

        Examples:
            Apply NMS to a set of boxes
            >>> boxes = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]])
            >>> scores = torch.tensor([0.9, 0.8])
            >>> keep = TorchNMS.nms(boxes, scores, 0.5)
        """
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)

        # Pre-allocate and extract coordinates once
        x1, y1, x2, y2 = boxes.unbind(1)
        areas = (x2 - x1) * (y2 - y1)

        # Sort by scores descending
        order = scores.argsort(0, descending=True)

        # Pre-allocate keep list with maximum possible size
        keep = torch.zeros(order.numel(), dtype=torch.int64, device=boxes.device)
        keep_idx = 0
        while order.numel() > 0:
            i = order[0]
            keep[keep_idx] = i
            keep_idx += 1

            if order.numel() == 1:
                break
            # Vectorized IoU calculation for remaining boxes
            rest = order[1:]
            xx1 = torch.maximum(x1[i], x1[rest])
            yy1 = torch.maximum(y1[i], y1[rest])
            xx2 = torch.minimum(x2[i], x2[rest])
            yy2 = torch.minimum(y2[i], y2[rest])

            # Fast intersection and IoU
            w = (xx2 - xx1).clamp_(min=0)
            h = (yy2 - yy1).clamp_(min=0)
            inter = w * h
            # Early exit: skip IoU calculation if no intersection
            if inter.sum() == 0:
                # No overlaps with current box, keep all remaining boxes
                order = rest
                continue
            iou = inter / (areas[i] + areas[rest] - inter)
            # Keep boxes with IoU <= threshold
            order = rest[iou <= iou_threshold]

        return keep[:keep_idx]

    @staticmethod
    def batched_nms(
        boxes: torch.Tensor,
        scores: torch.Tensor,
        idxs: torch.Tensor,
        iou_threshold: float,
        use_fast_nms: bool = False,
    ) -> torch.Tensor:
        """Batched NMS for class-aware suppression.

        Args:
            boxes (torch.Tensor): Bounding boxes with shape (N, 4) in xyxy format.
            scores (torch.Tensor): Confidence scores with shape (N,).
            idxs (torch.Tensor): Class indices with shape (N,).
            iou_threshold (float): IoU threshold for suppression.
            use_fast_nms (bool): Whether to use the Fast-NMS implementation.

        Returns:
            (torch.Tensor): Indices of boxes to keep after NMS.

        Examples:
            Apply batched NMS across multiple classes
            >>> boxes = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]])
            >>> scores = torch.tensor([0.9, 0.8])
            >>> idxs = torch.tensor([0, 1])
            >>> keep = TorchNMS.batched_nms(boxes, scores, idxs, 0.5)
        """
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)

        # Strategy: offset boxes by class index to prevent cross-class suppression
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]

        return (
            TorchNMS.fast_nms(boxes_for_nms, scores, iou_threshold)
            if use_fast_nms
            else TorchNMS.nms(boxes_for_nms, scores, iou_threshold)
        )


def _get_covariance_matrix(boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate covariance matrix from oriented bounding boxes.

    Args:
        boxes (torch.Tensor): A tensor of shape (N, 5) representing rotated bounding boxes, with xywhr format.

    Returns:
        (tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Covariance matrix components (a, b, c) where the covariance
            matrix is [[a, c], [c, b]], each of shape (N, 1).
    """
    # Gaussian bounding boxes, ignore the center points (the first two columns) because they are not needed here.
    gbbs = torch.cat((boxes[:, 2:4].pow(2) / 12, boxes[:, 4:]), dim=-1)
    a, b, c = gbbs.split(1, dim=-1)
    cos = c.cos()
    sin = c.sin()
    cos2 = cos.pow(2)
    sin2 = sin.pow(2)
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin


def batch_probiou(obb1: torch.Tensor | np.ndarray, obb2: torch.Tensor | np.ndarray, eps: float = 1e-7) -> torch.Tensor:
    """Calculate the probabilistic IoU between oriented bounding boxes.

    Args:
        obb1 (torch.Tensor | np.ndarray): A tensor of shape (N, 5) representing ground truth obbs, with xywhr format.
        obb2 (torch.Tensor | np.ndarray): A tensor of shape (M, 5) representing predicted obbs, with xywhr format.
        eps (float, optional): A small value to avoid division by zero.

    Returns:
        (torch.Tensor): A tensor of shape (N, M) representing obb similarities.

    References:
        https://arxiv.org/pdf/2106.06072v1.pdf
    """
    obb1 = torch.from_numpy(obb1) if isinstance(obb1, np.ndarray) else obb1
    obb2 = torch.from_numpy(obb2) if isinstance(obb2, np.ndarray) else obb2

    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = (x.squeeze(-1)[None] for x in obb2[..., :2].split(1, dim=-1))
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = (x.squeeze(-1)[None] for x in _get_covariance_matrix(obb2))

    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    return 1 - hd


def non_max_suppression_rotated(
    prediction,
    conf_thres: float = 0.001,
    iou_thres: float = 0.7,
    classes=None,
    agnostic: bool = False,
    multi_label: bool = True,  # False,
    labels=(),
    max_det: int = 300,
    nc: int = 80,  # 0,  # number of classes (optional)
    max_time_img: float = 0.05,
    max_nms: int = 30000,
    max_wh: int = 7680,
    rotated: bool = True,  # False,
    end2end: bool = True,  # False,
    return_idxs: bool = False,
):
    """Perform non-maximum suppression (NMS) on prediction results.

    Applies NMS to filter overlapping bounding boxes based on confidence and IoU thresholds. Supports multiple detection
    formats including standard boxes, rotated boxes, and masks.

    Args:
        prediction (torch.Tensor): Predictions with shape (batch_size, num_classes + 4 + num_masks, num_boxes)
            containing boxes, classes, and optional masks.
        conf_thres (float): Confidence threshold for filtering detections. Valid values are between 0.0 and 1.0.
        iou_thres (float): IoU threshold for NMS filtering. Valid values are between 0.0 and 1.0.
        classes (list[int], optional): List of class indices to consider. If None, all classes are considered.
        agnostic (bool): Whether to perform class-agnostic NMS.
        multi_label (bool): Whether each box can have multiple labels.
        labels (list[list[Union[int, float, torch.Tensor]]]): A priori labels for each image.
        max_det (int): Maximum number of detections to keep per image.
        nc (int): Number of classes. Indices after this are considered masks.
        max_time_img (float): Maximum time in seconds for processing one image.
        max_nms (int): Maximum number of boxes for NMS.
        max_wh (int): Maximum box width and height in pixels.
        rotated (bool): Whether to handle Oriented Bounding Boxes (OBB).
        end2end (bool): Whether the model is end-to-end and doesn't require NMS. # for seg
        return_idxs (bool): Whether to return the indices of kept detections.

    Returns:
        output (list[torch.Tensor]): List of detections per image with shape (num_boxes, 6 + num_masks) containing (x1,
            y1, x2, y2, confidence, class, mask1, mask2, ...).
        keepi (list[torch.Tensor]): Indices of kept detections if return_idxs=True.
    """
    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output
    if classes is not None:
        classes = torch.tensor(classes, device=prediction.device)

    if prediction.shape[-1] == 6 or end2end:  # end-to-end model (BNC, i.e. 1,300,6)
        output = [pred[pred[:, 4] > conf_thres][:max_det] for pred in prediction]
        if classes is not None:
            output = [pred[(pred[:, 5:6] == classes).any(1)] for pred in output]
        return output

    bs = prediction.shape[0]  # batch size (BCN, i.e. 1,84,6300)
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    extra = prediction.shape[1] - nc - 4  # number of extra info
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates
    xinds = torch.arange(prediction.shape[-1], device=prediction.device).expand(bs, -1)[..., None]  # to track idxs

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    time_limit = 2.0 + max_time_img * bs  # seconds to quit after
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
    if not rotated:
        prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy

    t = time.time()
    output = [torch.zeros((0, 6 + extra), device=prediction.device)] * bs
    keepi = [torch.zeros((0, 1), device=prediction.device)] * bs  # to store the kept idxs
    for xi, (x, xk) in enumerate(zip(prediction, xinds)):  # image index, (preds, preds indices)
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        filt = xc[xi]  # confidence
        x = x[filt]
        if return_idxs:
            xk = xk[filt]

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]) and not rotated:
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + extra + 4), device=x.device)
            v[:, :4] = xywh2xyxy(lb[:, 1:5])  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, extra), 1)

        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
            if return_idxs:
                xk = xk[i]
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            filt = conf.view(-1) > conf_thres
            x = torch.cat((box, conf, j.float(), mask), 1)[filt]
            if return_idxs:
                xk = xk[filt]

        # Filter by class
        if classes is not None:
            filt = (x[:, 5:6] == classes).any(1)
            x = x[filt]
            if return_idxs:
                xk = xk[filt]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            filt = x[:, 4].argsort(descending=True)[:max_nms]  # sort by confidence and remove excess boxes
            x = x[filt]
            if return_idxs:
                xk = xk[filt]

        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        scores = x[:, 4]  # scores
        if rotated:
            boxes = torch.cat((x[:, :2] + c, x[:, 2:4], x[:, -1:]), dim=-1)  # xywhr
            i = TorchNMS.fast_nms(boxes, scores, iou_thres, iou_func=batch_probiou)
        else:
            boxes = x[:, :4] + c  # boxes (offset by class)
            # Speed strategy: torchvision for val or already loaded (faster), TorchNMS for predict (lower latency)
            i = torchvision.ops.nms(boxes, scores, iou_thres)
        i = i[:max_det]  # limit detections

        output[xi] = x[i]
        if return_idxs:
            keepi[xi] = xk[i].view(-1)
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return (output, keepi) if return_idxs else output


# # Note: Temporary workaround for the mismatch in output tensor order between the original ONNX model and DXNN.
# #       The _wrapper function will be removed once the issue is properly fixed.
# def find_index_from_tensors_name(data, name):
#     indices = [i for i, item in enumerate(data) if name in item['name']]
#     if indices.__len__() != 1:
#         raise Exception(f"Expected exactly one output tensor, but found a different number,
#                         num of output tensor: {indices.__len__()}")
#     else:
#         return indices[0]

# def ssd_nms_wrapper(
#     outputs: List[np.ndarray],
#     prob_threshold: float = 0.01,
#     iou_threshold: float = 0.45,
#     session=None):
#     if session:
#         if session.type == SessionType.onnxruntime:
#             pass
#         elif session.type == SessionType.dxruntime:
#             output_tensors_info = session.inference_engine.get_output_tensors_info()
#             scores_idx = find_index_from_tensors_name(output_tensors_info, "scores")
#             boxes_idx = find_index_from_tensors_name(output_tensors_info, "boxes")
#             outputs = [outputs[scores_idx], outputs[boxes_idx]]
#         else:
#             raise Exception(f"Invalid SeessionType: {session.type}")
#     else:
#         pass

#     return ssd_nms(outputs, prob_threshold, iou_threshold)


def find_index_from_tensors_name(data, name):
    indices = [i for i, item in enumerate(data) if name in item["name"]]
    if indices.__len__() != 1:
        raise Exception(
            "Expected exactly one output tensor, but found a different number, "
            f"num of output tensor: {indices.__len__()}"
        )
    else:
        return indices[0]


# Note: Temporary workaround for the mismatch in output tensor order between the original ONNX model and DXNN.
#       The _wrapper function will be removed once the issue is properly fixed.
def ssd_nms_wrapper(outputs: List[np.ndarray], prob_threshold: float = 0.01, iou_threshold: float = 0.45, session=None):
    if session:
        if session.type == SessionType.onnxruntime:
            pass
        elif session.type == SessionType.dxruntime:
            output_tensors_info = session.inference_engine.get_output_tensors_info()
            scores_idx = find_index_from_tensors_name(output_tensors_info, "scores")
            boxes_idx = find_index_from_tensors_name(output_tensors_info, "boxes")
            outputs = [outputs[scores_idx], outputs[boxes_idx]]
        else:
            raise Exception(f"Invalid SeessionType: {session.type}")
    else:
        pass

    return ssd_nms(outputs, prob_threshold, iou_threshold)


def ssd_nms(outputs: List[np.ndarray], prob_threshold: float = 0.01, iou_threshold: float = 0.45):
    batched_class_scores, batched_boxes_coordinates = outputs
    batched_class_scores, batched_boxes_coordinates = torch.from_numpy(batched_class_scores), torch.from_numpy(
        batched_boxes_coordinates
    )

    num_classes = batched_class_scores.size(2)
    batched_picked_boxes, batched_picked_scores, batched_picked_classes = [], [], []
    for class_scores, boxes_coordinates in zip(batched_class_scores, batched_boxes_coordinates):
        picked_boxes, picked_scores, picked_classes = [], [], []

        for class_idx in range(1, num_classes):
            if class_idx == 0:
                continue
            mask = class_scores[..., class_idx] > prob_threshold

            filtered_score = class_scores[mask, class_idx]  # [num_filterd]
            filtered_boxes = boxes_coordinates[mask, :]  # [num_filterd, 4]

            if filtered_boxes.size(0) == 0:
                continue

            nms_boxes, nms_scores = hard_nms(filtered_boxes, filtered_score, iou_threshold)
            nms_classes = torch.zeros_like(nms_scores) + class_idx

            picked_boxes.append(nms_boxes)
            picked_scores.append(nms_scores)
            picked_classes.append(nms_classes)

        batched_picked_boxes.append(torch.cat(picked_boxes))
        batched_picked_scores.append(torch.cat(picked_scores))
        batched_picked_classes.append(torch.cat(picked_classes))
    return batched_picked_boxes, batched_picked_scores, batched_picked_classes


def hard_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float,
    candidate_size: int = 200,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _, indexes = scores.sort(descending=True)
    indexes = indexes[:candidate_size]
    picked = []
    while len(indexes) > 0:
        selected_idx = indexes[0]
        picked.append(selected_idx)

        selected_box = boxes[None, selected_idx, :]  # [1, 4]

        # remove selected_idx from indexes
        indexes = indexes[1:]

        rest_boxes = boxes[indexes, :]

        ious = calculate_iou(rest_boxes, selected_box)
        indexes = indexes[ious <= iou_threshold]
    return boxes[picked, :], scores[picked, ...]


def non_maximum_suppression_for_pose(
    outputs: List[np.ndarray | torch.Tensor],
    conf_thres: float = 0.001,
    iou_thres: float = 0.7,
    max_output_boxes: int = 300,
    num_classes: int | None = None,
    agnostic: bool = False,
) -> torch.Tensor:
    """
    Work-around NMS function for YOLOv8 pose models.
    Original function is not sutable because of num classes value.
    """

    if not isinstance(outputs, torch.Tensor):
        outputs = torch.from_numpy(outputs[0])

    score_index = 1 if num_classes is None or num_classes == 0 else num_classes
    conf_score = outputs[..., 4 : 4 + score_index].amax(2)
    mask = conf_score > conf_thres
    extra = outputs.shape[-1] - 4 - score_index

    nms_outputs = []
    for batch, output in enumerate(outputs):
        output = output[mask[batch]]
        boxes, classes_score, extras = output.split((4, score_index, extra), 1)

        conf, j = classes_score.max(1, keepdim=True)
        filt = conf.view(-1) > conf_thres
        filtered_output = torch.cat((boxes, conf, j, extras), 1)[filt]

        # If none remain process next image
        if filtered_output.numel() == 0:
            continue

        # Check shape and apply max_nms limit (Ultralytics logic)
        n = filtered_output.shape[0]  # number of boxes
        if n > MAX_NMS:  # excess boxes
            # Sort by confidence and remove excess boxes
            filt = filtered_output[:, 4].argsort(descending=True)[:MAX_NMS]
            filtered_output = filtered_output[filt]

        # Convert boxes from cxcywh to xyxy format
        boxes = filtered_output[..., :4]
        scores = filtered_output[..., 4, None]
        class_indices = filtered_output[..., 5, None]

        # Convert cxcywh to xyxy format BEFORE NMS
        boxes_xyxy = convert_cxcywh_2_xyxy(boxes)

        # Batched NMS (Ultralytics logic)
        # Apply class offset for NMS (0 if agnostic, else class * MAX_WH)
        c = class_indices * (0 if agnostic else MAX_WH)
        boxes_for_nms = boxes_xyxy + c

        # Apply NMS
        nms_output_index = torchvision.ops.nms(boxes_for_nms, scores[..., 0], iou_thres)

        # Limit detections to max_output_boxes
        num_nms_outputs = nms_output_index.size(0)
        if num_nms_outputs > max_output_boxes:
            nms_output_index = nms_output_index[:max_output_boxes]

        # Rebuild final output with extras
        # Format: [xyxy, scores, class_indices, extras...]
        final_output = torch.cat([boxes_xyxy, scores, class_indices], dim=1)
        if filtered_output.shape[1] > 6:  # If we have extras
            final_output = torch.cat([final_output, filtered_output[:, 6:]], dim=1)

        processed_output = final_output[nms_output_index]
        nms_outputs.append(processed_output)

    if len(nms_outputs) == 0:
        return torch.empty((0, 0))
    return torch.concat(nms_outputs, axis=0)


def ssd_vgg16_postprocessing(
    outputs: List[np.ndarray],
    prob_threshold: float = 0.001,
    iou_threshold: float = 0.7,
    input_size: int = 300,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    SSD VGG16 postprocessing for VOC detection (21 classes).

    Outputs:
    - scores: [batch, 8732, 21] - class scores (including background class 0)
    - boxes: [batch, 8732, 4] - box coordinates in normalized format

    Returns:
        Same format as ssd_nms:
        (batched_picked_boxes, batched_picked_scores, batched_picked_classes)
    """
    # Find scores and boxes by shape
    scores_out = None
    boxes_out = None

    for out in outputs:
        if out.shape[-1] == 21:
            scores_out = out
        elif out.shape[-1] == 4:
            boxes_out = out

    if scores_out is None or boxes_out is None:
        return [torch.empty((0, 4))], [torch.empty((0,))], [torch.empty((0,))]

    batched_class_scores = torch.from_numpy(scores_out)
    batched_boxes_coordinates = torch.from_numpy(boxes_out)

    num_classes = batched_class_scores.size(2)  # 21
    batched_picked_boxes, batched_picked_scores, batched_picked_classes = [], [], []

    for class_scores, boxes_coordinates in zip(batched_class_scores, batched_boxes_coordinates):
        picked_boxes, picked_scores, picked_classes = [], [], []

        # Process each class (skip background class 0)
        for class_idx in range(1, num_classes):
            mask = class_scores[:, class_idx] > prob_threshold

            filtered_score = class_scores[mask, class_idx]
            filtered_boxes = boxes_coordinates[mask, :]

            if filtered_boxes.size(0) == 0:
                continue

            # SSD VGG16 outputs are already in xyxy format normalized (0-1)
            x1 = filtered_boxes[:, 0].clamp(0, 1)
            y1 = filtered_boxes[:, 1].clamp(0, 1)
            x2 = filtered_boxes[:, 2].clamp(0, 1)
            y2 = filtered_boxes[:, 3].clamp(0, 1)

            boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)

            # For NMS, use pixel coordinates temporarily
            nms_boxes_pixel = boxes_xyxy * input_size
            nms_boxes, nms_scores = hard_nms(nms_boxes_pixel, filtered_score, iou_threshold)

            # Convert back to normalized coords for evaluator
            nms_boxes = nms_boxes / input_size
            nms_classes = torch.zeros_like(nms_scores) + class_idx

            picked_boxes.append(nms_boxes)
            picked_scores.append(nms_scores)
            picked_classes.append(nms_classes)

        if len(picked_boxes) > 0:
            batched_picked_boxes.append(torch.cat(picked_boxes))
            batched_picked_scores.append(torch.cat(picked_scores))
            batched_picked_classes.append(torch.cat(picked_classes))
        else:
            batched_picked_boxes.append(torch.empty((0, 4)))
            batched_picked_scores.append(torch.empty((0,)))
            batched_picked_classes.append(torch.empty((0,)))

    return batched_picked_boxes, batched_picked_scores, batched_picked_classes
