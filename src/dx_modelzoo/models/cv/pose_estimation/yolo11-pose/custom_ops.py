"""YOLO11 pose estimation postprocessing custom ops."""
from __future__ import annotations

import numpy as np

from dx_modelzoo.postprocessing import POSTPROCESSING_REGISTRY
from dx_modelzoo.postprocessing.coord_scaler import unpad_and_scale
from dx_modelzoo.postprocessing.decode_utils import cxcywh_to_xyxy
from dx_modelzoo.postprocessing.nms import MAX_WH, nms_numpy


def _empty_pose_result(num_keypoints=17):
    return (
        np.empty((0, 4), dtype=np.float64),
        np.empty(0, dtype=np.float64),
        np.empty(0, dtype=np.float64),
        np.empty((0, num_keypoints, 3), dtype=np.float64),
    )


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


@POSTPROCESSING_REGISTRY.register("yolo11_pose_decode")
class YoloV11PoseDecode:
    """YOLO11 Pose: flexible layout, auto-detect class count."""

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
        if raw.ndim == 3:
            raw = raw[0]
        if raw.shape[0] < raw.shape[1]:
            raw = raw.T
        if raw.shape[0] == 0 or raw.shape[1] == 0:
            return _empty_pose_result(self.num_keypoints)

        n_cols = raw.shape[1]
        remaining = n_cols - 4
        num_classes = max(1, remaining - self.num_keypoints * 3)

        boxes = raw[:, :4]
        class_scores = raw[:, 4 : 4 + num_classes]
        kpt_data = raw[:, 4 + num_classes :]

        scores = class_scores.max(axis=1)
        cls_ids = class_scores.argmax(axis=1)

        mask = scores > self.conf_thres
        if not mask.any():
            return _empty_pose_result(self.num_keypoints)

        boxes = cxcywh_to_xyxy(boxes[mask])
        s = scores[mask]
        c = cls_ids[mask]
        kpts = kpt_data[mask]

        keep = nms_numpy(boxes + (c[:, None].astype(np.float64) * MAX_WH), s, self.iou_thres)
        boxes = boxes[keep]
        s = s[keep]
        c = c[keep]
        kpts = kpts[keep]

        n_kpts = kpts.shape[1] // 3
        keypoints = kpts.reshape(-1, n_kpts, 3)

        result = (boxes, s, c.astype(np.float64), keypoints)
        return _rescale_pose(result, pad_resize=self.pad_resize, **kwargs)
