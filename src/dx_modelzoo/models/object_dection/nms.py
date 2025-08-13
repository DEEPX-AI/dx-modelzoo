from typing import List, Tuple

import numpy as np
import torch
import torchvision

from dx_modelzoo.enums import SessionType
from dx_modelzoo.utils.detection import calculate_iou, convert_cxcywh_2_xyxy

MAX_WH = 7680
MAX_NMS = 30000


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
    iou_thres: float = 0.6,
    max_output_boxes: int = 300,
    multi_label: bool = False,
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
    if not isinstance(outputs, torch.Tensor):
        outputs = torch.from_numpy(outputs[0]).clone()
    mask = outputs[..., 4] > conf_thres
    nms_outputs = []
    for batch, output in enumerate(outputs):
        output = output[mask[batch]]
        confidence_scores = get_confidence_scores(output)
        filtered_output = filter_outputs_by_conf_score(output, confidence_scores, conf_thres, multi_label)
        boxes = convert_cxcywh_2_xyxy(filtered_output[..., :4])
        scores = filtered_output[..., 4, None]
        class_indeices = filtered_output[..., 5, None]

        num_boxes = boxes.size(0)
        sorted_mask = scores[..., 0].argsort(descending=True)
        if num_boxes > MAX_NMS:
            sorted_mask = sorted_mask[:MAX_NMS]

        boxes = boxes[sorted_mask]
        scores = scores[sorted_mask]
        class_indeices = class_indeices[sorted_mask]

        nms_output_index = torchvision.ops.nms(boxes + (class_indeices * MAX_WH), scores[..., 0], iou_thres)

        num_nms_outputs = nms_output_index.size(0)
        if num_nms_outputs > max_output_boxes:
            nms_output_index = nms_output_index[:max_output_boxes]

        processed_output = torch.cat((boxes, scores, class_indeices), dim=1)
        nms_outputs.append(processed_output[nms_output_index])

    return torch.concat(nms_outputs, axis=0)


def non_maximum_suppression2(
    outputs: List[np.ndarray | torch.Tensor],
    conf_thres: float = 0.001,
    iou_thres: float = 0.6,
    max_output_boxes: int = 300,
) -> torch.Tensor:
    if not isinstance(outputs, torch.Tensor):
        outputs = torch.from_numpy(outputs[0]).clone()

    num_classes = outputs.shape[2] - 4
    conf_score = outputs[..., 4:].amax(2)
    mask = conf_score > conf_thres

    nms_outputs = []
    for batch, output in enumerate(outputs):
        output = output[mask[batch]]

        boxes, classes_score = output.split((4, num_classes), 1)

        box_index, class_index = torch.where(classes_score > conf_thres)
        filtered_output = torch.cat(
            (
                boxes[box_index],
                classes_score[box_index, class_index, None],
                class_index[:, None].float(),
            ),
            1,
        )
        boxes = convert_cxcywh_2_xyxy(filtered_output[..., :4])
        scores = filtered_output[..., 4, None]
        class_indeices = filtered_output[..., 5, None]

        num_boxes = boxes.size(0)
        sorted_mask = scores[..., 0].argsort(descending=True)
        if num_boxes > MAX_NMS:
            sorted_mask = sorted_mask[:MAX_NMS]

        boxes = boxes[sorted_mask]
        scores = scores[sorted_mask]
        class_indeices = class_indeices[sorted_mask]

        nms_output_index = torchvision.ops.nms(boxes + (class_indeices * MAX_WH), scores[..., 0], iou_thres)

        num_nms_outputs = nms_output_index.size(0)
        if num_nms_outputs > max_output_boxes:
            nms_output_index = nms_output_index[:max_output_boxes]

        processed_output = torch.cat((boxes, scores, class_indeices), dim=1)
        nms_outputs.append(processed_output[nms_output_index])

    return torch.concat(nms_outputs, axis=0)

# # Note: Temporary workaround for the mismatch in output tensor order between the original ONNX model and DXNN.
# #       The _wrapper function will be removed once the issue is properly fixed.
# def find_index_from_tensors_name(data, name):
#     indices = [i for i, item in enumerate(data) if name in item['name']]
#     if indices.__len__() != 1:
#         raise Exception(f"Expected exactly one output tensor, but found a different number, num of output tensor: {indices.__len__()}")
#     else:
#         return indices[0]

# def ssd_nms_wrapper(outputs: List[np.ndarray], prob_threshold: float = 0.01, iou_threshold: float = 0.45, session=None):
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
