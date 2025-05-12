from typing import List, Tuple

import numpy as np
import torch


def get_ratios(image: np.ndarray, origin_shape: List[torch.Tensor]) -> torch.Tensor:
    """get image's heigh, width ratio for resize.

    Args:
        image (np.ndarray): resized image.
        origin_shape (List[torch.Tensor]): origin image shape.

    Returns:
        Tuple[float, float]: height, width raitos.
    """
    origin_height, origin_width, _ = origin_shape
    size = image.shape[3]

    ratios = size / torch.maximum(origin_height, origin_width)
    return ratios


def get_pad_size(
    image: np.ndarray, origin_shape: List[torch.Tensor], ratios: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """get pad size of resized image.

    Args:
        image (np.ndarray): resized image.
        origin_shape (List[torch.Tensor]): origin input shape.
        ratios (Tuple[int, int]): ratios.

    Returns:
        Tuple[float, float]: height, width pad size.
    """
    origin_height, origin_width, _ = origin_shape
    size = image.shape[3]
    new_height = (origin_height * ratios).type(torch.int)
    new_width = (origin_width * ratios).type(torch.int)
    height_pad = (size - new_height) / 2
    width_pad = (size - new_width) / 2
    return height_pad, width_pad


def scale_boxes(
    boxes: torch.Tensor,
    origin_shape: List[torch.Tensor],
    ratios: torch.Tensor,
    pads: Tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """scaling boxes to origin shape.

    Args:
        boxes (np.ndarray): model output boxes. boxes format is xyxy format.
        origin_shape (List[torch.Tensor]): origin image shape.
        ratios (Tuple[float, float]): ratios for resize.
        pads (Tuple[float, float]): pad sizes for resize.

    Returns:
        np.ndarray: scaled boxes.
    """
    boxes[:, [0, 2]] -= pads[1] - 0.1  # x padding
    boxes[:, [1, 3]] -= pads[0] - 0.1  # y padding
    boxes[:, :4] /= ratios
    return clip_boxes(boxes, origin_shape)


def clip_boxes(boxes: torch.Tensor, origin_shape: List[torch.Tensor]) -> torch.Tensor:
    """clipping boxes coordinates.
    if box's cordinates is over or under the origin shapes, clipping it.

    Args:
        boxes (np.ndarray): input boxes. box format is xyxy.
        origin_shape (List[torch.Tensor]): origin image shape.

    Returns:
        np.ndarray: cliped boxes.
    """
    boxes[:, 0].clamp_(0, origin_shape[1])  # x1
    boxes[:, 1].clamp_(0, origin_shape[0])  # y1
    boxes[:, 2].clamp_(0, origin_shape[1])  # x2
    boxes[:, 3].clamp_(0, origin_shape[0])  # y2
    return boxes


def scale_landmarks(
    landmarks: torch.Tensor,
    origin_shape: List[torch.Tensor],
    ratios: torch.Tensor,
    pads: Tuple[torch.Tensor, torch.Tensor],
):

    landmarks[:, [0, 2, 4, 6, 8]] -= pads[0] - 0.1  # x padding
    landmarks[:, [1, 3, 5, 7, 9]] -= pads[1] - 0.1  # y padding
    landmarks[:, :10] /= ratios
    return clip_landmarks(landmarks, origin_shape)


def clip_landmarks(landmarks: torch.Tensor, origin_shape: List[torch.Tensor]):
    landmarks[:, 0].clamp_(0, origin_shape[1])  # x1
    landmarks[:, 1].clamp_(0, origin_shape[0])  # y1
    landmarks[:, 2].clamp_(0, origin_shape[1])  # x2
    landmarks[:, 3].clamp_(0, origin_shape[0])  # y2
    landmarks[:, 4].clamp_(0, origin_shape[1])  # x3
    landmarks[:, 5].clamp_(0, origin_shape[0])  # y3
    landmarks[:, 6].clamp_(0, origin_shape[1])  # x4
    landmarks[:, 7].clamp_(0, origin_shape[0])  # y4
    landmarks[:, 8].clamp_(0, origin_shape[1])  # x5
    landmarks[:, 9].clamp_(0, origin_shape[0])  # y5
    return landmarks


def convert_cxcywh_2_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """convert cxcywh box format to xyxy format.

    Args:
        boxes (torch.Tensor): cxcywh box format.

    Returns:
        torch.Tensor: converted boxes.
    """
    converted_boxes = boxes.clone()

    converted_boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # top left x
    converted_boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # top left y
    converted_boxes[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # bottom right x
    converted_boxes[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # bottom right y
    return converted_boxes


def convert_xyxy_2_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """converted xyxy boxes to cxcywh boxes.

    Args:
        boxes (np.ndarray): xyxy boxes.

    Returns:
        np.ndarray: converted boxes.
    """
    converted_boxes = boxes.clone()
    converted_boxes[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2  # x center
    converted_boxes[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2  # y center
    converted_boxes[:, 2] = boxes[:, 2] - boxes[:, 0]  # width
    converted_boxes[:, 3] = boxes[:, 3] - boxes[:, 1]  # height
    converted_boxes[:, :2] -= converted_boxes[:, 2:] / 2  # xy center to left corner
    return converted_boxes


def calculate_iou(boxes0: torch.Tensor, boxes1: torch.Tensor, eps: float = 1e-5):
    overlap_left_top = torch.max(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = torch.min(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = calculate_box_area(overlap_left_top, overlap_right_bottom)
    area0 = calculate_box_area(boxes0[..., :2], boxes0[..., 2:])
    area1 = calculate_box_area(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def calculate_box_area(left_top: torch.Tensor, right_bottom: torch.Tensor) -> torch.Tensor:
    hw = torch.clamp(right_bottom - left_top, min=0.0)
    return hw[..., 0] * hw[..., 1]
