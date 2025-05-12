import torch
import torchvision

from dx_modelzoo.utils.detection import convert_cxcywh_2_xyxy


def get_confidence_scores_for_yolov5(output: torch.Tensor) -> torch.Tensor:
    """caculate confidence score
    confidence scores is object confidence socres * class confidence scores.
    Args:
        output (torch.Tensor): confidence score.

    Returns:
        torch.Tensor: confidenc scores.
    """
    class_conf_scores = output[..., 4, None]
    obj_conf_scores = output[..., -1, None]
    return obj_conf_scores * class_conf_scores


def non_max_suppression_for_yolv5_face(outputs: torch.Tensor, conf_threshold: float, iou_threshold: float):
    # yolov5's num_landmark is 5, num_classes 1

    nms_outputs = []
    mask = outputs[..., 4] > conf_threshold
    for batch, output in enumerate(outputs):
        output = output[mask[batch]]
        confidence_scores = get_confidence_scores_for_yolov5(output)  # [num_boxes, 1]
        boxes = convert_cxcywh_2_xyxy(output[..., :4])
        filterd_output = torch.cat((boxes, confidence_scores, output[..., 5:15]), 1)[
            confidence_scores[..., 0] > conf_threshold
        ]

        nms_output_index = torchvision.ops.nms(filterd_output[..., :4], filterd_output[..., 4], iou_threshold)
        nms_outputs.append(filterd_output[nms_output_index])
    return torch.concat(nms_outputs, axis=0)


def get_confidence_scores_for_yolov7(output: torch.Tensor) -> torch.Tensor:
    """caculate confidence score
    confidence scores is object confidence socres * class confidence scores.
    Args:
        output (torch.Tensor): confidence score.

    Returns:
        torch.Tensor: confidenc scores.
    """
    class_conf_scores = output[..., 4, None]
    obj_conf_scores = output[..., 5, None]
    return obj_conf_scores * class_conf_scores


def non_max_suppression_for_yolv7_face(outputs: torch.Tensor, conf_threshold: float, iou_threshold: float):
    nms_outputs = []
    mask = outputs[..., 4] > conf_threshold
    for batch, output in enumerate(outputs):
        output = output[mask[batch]]
        confidence_scores = get_confidence_scores_for_yolov7(output)

        boxes = convert_cxcywh_2_xyxy(output[..., :4])
        filterd_output = torch.cat((boxes, confidence_scores, output[..., 5:15]), 1)[
            confidence_scores[..., 0] > conf_threshold
        ]
        nms_output_index = torchvision.ops.nms(filterd_output[..., :4], filterd_output[..., 4], iou_threshold)
        nms_outputs.append(filterd_output[nms_output_index])

    return torch.concat(nms_outputs, axis=0)
