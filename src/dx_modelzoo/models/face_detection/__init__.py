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


def non_max_suppression_for_scrfd(prediction, conf_thres=0.25, iou_thres=0.45):
    xc = prediction[..., 4] > conf_thres  # candidates

    output = [torch.zeros((0, 16), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence

        n = x.shape[0]  # number of boxes
        if not n:
            continue

        boxes, scores = x[:, :4], x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        output[xi] = x[i]

    return output


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def scale_coords_for_yolov5_face(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords
