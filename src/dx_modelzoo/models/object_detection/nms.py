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
    cxcywh2xyxy_conversion: bool = True,
) -> torch.Tensor:
    if not isinstance(outputs, torch.Tensor):
        outputs = torch.from_numpy(outputs[0])

    num_classes = outputs.shape[2] - 4

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
    iou_thres: float = 0.6,
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
