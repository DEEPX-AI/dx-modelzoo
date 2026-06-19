"""YOLOv8/v11 pose estimation postprocessing custom ops."""
from __future__ import annotations

import numpy as np

from dx_modelzoo.postprocessing import POSTPROCESSING_REGISTRY
from dx_modelzoo.postprocessing.coord_scaler import unpad_and_scale
from dx_modelzoo.postprocessing.decode_utils import cxcywh_to_xyxy
from dx_modelzoo.postprocessing.nms import MAX_NMS, MAX_WH, nms_numpy


def _nms_for_pose(
    outputs: np.ndarray,
    conf_thres: float = 0.25,
    iou_thres: float = 0.7,
    max_output_boxes: int = 300,
    num_classes: int = 1,
) -> np.ndarray:
    """NMS for pose models. Returns [M, 6+K*3] (boxes, conf, cls, keypoints)."""
    if outputs.ndim == 2:
        outputs = outputs[np.newaxis, ...]

    all_results = []
    for batch_idx in range(outputs.shape[0]):
        output = outputs[batch_idx]
        if output.shape[0] == 0:
            continue

        boxes_cxcywh = output[:, :4]
        if num_classes == 0:
            scores = output[:, 4]
            cls_ids = np.zeros(len(output), dtype=np.int64)
            extras = output[:, 5:]
        else:
            class_scores = output[:, 4 : 4 + num_classes]
            scores = class_scores.max(axis=1)
            cls_ids = class_scores.argmax(axis=1)
            extras = output[:, 4 + num_classes :]

        mask = scores > conf_thres
        if not mask.any():
            continue

        boxes = cxcywh_to_xyxy(boxes_cxcywh[mask])
        s = scores[mask]
        c = cls_ids[mask]
        e = extras[mask]

        if len(boxes) > MAX_NMS:
            top = np.argsort(s)[-MAX_NMS:]
            boxes, s, c, e = boxes[top], s[top], c[top], e[top]

        offset_boxes = boxes + (c[:, None].astype(np.float64) * MAX_WH)
        keep = nms_numpy(offset_boxes, s, iou_thres)
        if len(keep) > max_output_boxes:
            keep = keep[:max_output_boxes]

        result = np.column_stack([boxes[keep], s[keep], c[keep].astype(np.float64), e[keep]])
        all_results.append(result)

    if not all_results:
        return np.empty((0, 6), dtype=np.float64)
    return np.concatenate(all_results, axis=0)


def _empty_pose_result(num_keypoints=17):
    return (
        np.empty((0, 4), dtype=np.float64),
        np.empty(0, dtype=np.float64),
        np.empty(0, dtype=np.float64),
        np.empty((0, num_keypoints, 3), dtype=np.float64),
    )


def _format_pose_result(kept, num_keypoints=17):
    if kept.shape[0] == 0:
        return _empty_pose_result(num_keypoints)
    boxes = kept[:, :4]
    scores = kept[:, 4]
    cls_ids = kept[:, 5]
    extras = kept[:, 6:]
    n_kpts = extras.shape[1] // 3
    keypoints = extras.reshape(-1, n_kpts, 3)
    return boxes, scores, cls_ids, keypoints


def _rescale_pose(result, pad_resize=True, **kwargs):
    origin_hw = kwargs.get("origin_hw")
    input_hw = kwargs.get("input_hw")
    if origin_hw is None or input_hw is None:
        return result
    boxes, scores, cls_ids, keypoints = result
    if len(boxes) == 0:
        return result
    boxes = boxes.copy()
    boxes[:, :4] = unpad_and_scale(boxes[:, :4], input_hw, origin_hw, pad_resize=pad_resize)
    if keypoints is not None and len(keypoints) > 0:
        keypoints = keypoints.copy()
        kp_boxes = keypoints[:, :, :2].reshape(-1, 2)
        pseudo = np.column_stack([kp_boxes, kp_boxes])
        scaled = unpad_and_scale(pseudo, input_hw, origin_hw, pad_resize=pad_resize)
        keypoints[:, :, :2] = scaled[:, :2].reshape(keypoints.shape[0], -1, 2)
    return boxes, scores, cls_ids, keypoints


@POSTPROCESSING_REGISTRY.register("yolov8_pose_decode")
class YoloV8PoseDecode:
    """YOLOv8 Pose: [1, 56, N] → transpose → NMS → (boxes, scores, cls, kpts)."""

    def __init__(
        self,
        conf_thres: float = 0.25,
        iou_thres: float = 0.7,
        num_keypoints: int = 17,
        pad_resize: bool = True,
        **kwargs,
    ) -> None:
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.num_keypoints = num_keypoints
        self.pad_resize = pad_resize

    def __call__(self, outputs, **kwargs):
        raw = outputs[0] if isinstance(outputs, list) else outputs
        if raw.ndim == 2:
            raw = raw[np.newaxis, ...]
        if raw.ndim == 3 and raw.shape[1] == 4 + 1 + self.num_keypoints * 3:
            raw = np.transpose(raw, (0, 2, 1))
        kept = _nms_for_pose(raw, self.conf_thres, self.iou_thres, num_classes=0)
        result = _format_pose_result(kept, self.num_keypoints)
        return _rescale_pose(result, pad_resize=self.pad_resize, **kwargs)
