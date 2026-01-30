from typing import List

import numpy as np
import torch
import torchvision


def yolo_ppu_postprocessing(outputs: List[np.ndarray]):
    if outputs[0].shape[-1] == 0:
        return torch.zeros((0, 6), dtype=torch.float32)

    if outputs[0].shape[-1] != 32:
        outputs = outputs[0]
        outputs = torch.from_numpy(outputs)

        return outputs[0]

    MAX_WH = 7680
    MAX_NMS = 30000

    conf_thres = 0.001
    iou_thres = 0.7
    max_output_boxes = 300

    outputs = outputs[0]

    bboxes = outputs[:,:,:16].view(np.float32)
    scores = outputs[:,:,20:24].view(np.float32)
    labels = outputs[:,:,24:28].view(np.uint32).astype(np.float32)

    mask = scores > conf_thres
    mask = mask.squeeze(-1)

    # Apply mask to filter valid detections
    bboxes = torch.from_numpy(bboxes[mask])
    scores = torch.from_numpy(scores[mask])
    labels = torch.from_numpy(labels[mask])

    num_boxes = bboxes.shape[1]
    sorted_mask = scores[..., 0].argsort(descending=True)
    if num_boxes > MAX_NMS:
        sorted_mask = sorted_mask[:MAX_NMS]
    
    bboxes = bboxes[sorted_mask]
    scores = scores[sorted_mask]
    labels = labels[sorted_mask]
    
    nms_output_index = torchvision.ops.nms(bboxes + (labels * MAX_WH), scores[..., 0], iou_thres)

    num_nms_outputs = nms_output_index.size(0)
    if num_nms_outputs > max_output_boxes:
        nms_output_index = nms_output_index[:max_output_boxes]

    processed_output = torch.cat((bboxes, scores, labels), dim=1)

    result = processed_output[nms_output_index]

    return result