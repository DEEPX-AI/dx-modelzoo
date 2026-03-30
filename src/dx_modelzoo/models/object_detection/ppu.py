from typing import List, Literal

import numpy as np
import torch
import torchvision

MAX_WH = 7680
MAX_NMS = 30000


def yolo_ppu_postprocessing(
    outputs: List[np.ndarray],
    box_format: Literal["cxcywh", "xyxy"] = "xyxy",
    anchors=None,
    strides=None,
    yolo_version: Literal["v3", "v4", "v5", "v7", "x"] = "v7",
    conf_thres=0.001,
    iou_thres=0.7,
    max_det=300,
) -> torch.Tensor:
    outputs = outputs[0]

    if outputs.shape[-1] == 0:
        return torch.zeros((0, 6), dtype=torch.float32)

    if outputs.shape[-1] != 32:
        outputs = outputs
        outputs = torch.from_numpy(outputs)

        return outputs

    bboxes = outputs[:, :, :16].view(np.float32)
    scores = outputs[:, :, 20:24].view(np.float32)
    labels = outputs[:, :, 24:28].view(np.uint32).astype(np.float32)

    grid_info = outputs[0][:, 16:20].view(np.uint8)
    gY = grid_info[:, 0].astype(np.float32)
    gX = grid_info[:, 1].astype(np.float32)
    anchor_idx = grid_info[:, 2]
    layer_idx = grid_info[:, 3]

    if strides is None:
        strides = np.array([8, 16, 32, 64], dtype=np.float32)
    else:
        strides = np.array(strides, dtype=np.float32)

    # Filter out detections with invalid layer_idx (garbage PPU data)
    valid_mask = layer_idx < len(strides)
    if not np.all(valid_mask):
        bboxes = bboxes[:, valid_mask, :]
        scores = scores[:, valid_mask, :]
        labels = labels[:, valid_mask, :]
        gY = gY[valid_mask]
        gX = gX[valid_mask]
        anchor_idx = anchor_idx[valid_mask]
        layer_idx = layer_idx[valid_mask]

    if yolo_version:
        boxes = bboxes[0]
        stride = strides[layer_idx]

        if yolo_version == "x":
            # YOLOX (anchor-free): xy = (xy + grid) * stride, wh = exp(wh) * stride
            boxes_cx = (boxes[:, 0] + gX) * stride
            boxes_cy = (boxes[:, 1] + gY) * stride
            boxes_w = np.exp(boxes[:, 2]) * stride
            boxes_h = np.exp(boxes[:, 3]) * stride
        elif yolo_version in ["v3", "v4", "v5", "v7"]:
            if anchors:
                anchors_by_stride = {
                    s: np.array(anchors[i], dtype=np.float32) for i, s in enumerate(strides[: len(anchors)])
                }

                anchor_w = np.zeros(len(boxes), dtype=np.float32)
                anchor_h = np.zeros(len(boxes), dtype=np.float32)

                for s in anchors_by_stride.keys():
                    stride_mask = stride == s
                    if np.any(stride_mask):
                        anchors_arr = anchors_by_stride[s]
                        anchor_w[stride_mask] = anchors_arr[anchor_idx[stride_mask], 0]
                        anchor_h[stride_mask] = anchors_arr[anchor_idx[stride_mask], 1]

                # YOLOv3/v4/v5/v7: xy = (xy * 2 - 0.5 + grid) * stride, wh = (wh * 2)^2 * anchor
                boxes_cx = (boxes[:, 0] * 2.0 - 0.5 + gX) * stride
                boxes_cy = (boxes[:, 1] * 2.0 - 0.5 + gY) * stride
                boxes_w = (boxes[:, 2] ** 2 * 4.0) * anchor_w
                boxes_h = (boxes[:, 3] ** 2 * 4.0) * anchor_h
            else:
                raise ValueError(f"Anchors must be provided for YOLO version {yolo_version}")

        # (left, top, right, bottom)
        boxes_xyxy = np.column_stack(
            [
                boxes_cx - boxes_w * 0.5,
                boxes_cy - boxes_h * 0.5,
                boxes_cx + boxes_w * 0.5,
                boxes_cy + boxes_h * 0.5,
            ]
        )

        bboxes = torch.from_numpy(boxes_xyxy[np.newaxis, :, :]).float()
        scores = torch.from_numpy(scores).float()
        labels = torch.from_numpy(labels).float()
    elif yolo_version is None:
        # Convert to torch first
        bboxes = torch.from_numpy(bboxes).float()
        scores = torch.from_numpy(scores).float()
        labels = torch.from_numpy(labels).float()

        if box_format == "cxcywh":
            bboxes_xyxy = torch.zeros_like(bboxes)
            bboxes_xyxy[..., 0] = bboxes[..., 0] - bboxes[..., 2] / 2  # x1
            bboxes_xyxy[..., 1] = bboxes[..., 1] - bboxes[..., 3] / 2  # y1
            bboxes_xyxy[..., 2] = bboxes[..., 0] + bboxes[..., 2] / 2  # x2
            bboxes_xyxy[..., 3] = bboxes[..., 1] + bboxes[..., 3] / 2  # y2
            bboxes = bboxes_xyxy
        elif box_format == "xywh":
            bboxes_xyxy = torch.zeros_like(bboxes)
            bboxes_xyxy[..., 0] = bboxes[..., 0]
            bboxes_xyxy[..., 1] = bboxes[..., 1]
            bboxes_xyxy[..., 2] = bboxes[..., 0] + bboxes[..., 2]
            bboxes_xyxy[..., 3] = bboxes[..., 1] + bboxes[..., 3]
            bboxes = bboxes_xyxy
        elif box_format == "yxyx":
            bboxes_yxyx = torch.zeros_like(bboxes)
            bboxes_yxyx[..., 0] = bboxes[..., 1]  # y1
            bboxes_yxyx[..., 1] = bboxes[..., 0]  # x1
            bboxes_yxyx[..., 2] = bboxes[..., 3]  # y2
            bboxes_yxyx[..., 3] = bboxes[..., 2]  # x2
            bboxes = bboxes_yxyx
        elif box_format == "xyxy":
            pass
        else:
            raise ValueError(f"Unsupported box_format: {box_format}")
    else:
        raise ValueError(f"Unsupported YOLO version: {yolo_version}")

    # Apply confidence threshold
    mask = scores > conf_thres
    mask = mask.squeeze(-1)

    # Filter valid detections
    bboxes = bboxes[mask]
    scores = scores[mask]
    labels = labels[mask]

    if bboxes.shape[0] == 0:
        return torch.zeros((0, 6), dtype=torch.float32)

    num_boxes = bboxes.shape[0]
    sorted_mask = scores[..., 0].argsort(descending=True)
    if num_boxes > MAX_NMS:
        sorted_mask = sorted_mask[:MAX_NMS]

    bboxes = bboxes[sorted_mask]
    scores = scores[sorted_mask]
    labels = labels[sorted_mask]

    nms_output_index = torchvision.ops.nms(bboxes + (labels * MAX_WH), scores[..., 0], iou_thres)

    num_nms_outputs = nms_output_index.size(0)
    if num_nms_outputs > max_det:
        nms_output_index = nms_output_index[:max_det]

    processed_output = torch.cat((bboxes, scores, labels), dim=1)

    result = processed_output[nms_output_index]

    return result
