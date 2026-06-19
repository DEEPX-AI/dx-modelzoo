"""YOLOv5 pose estimation postprocessing custom ops."""
from __future__ import annotations

import numpy as np

from dx_modelzoo.postprocessing import POSTPROCESSING_REGISTRY
from dx_modelzoo.postprocessing.coord_scaler import unpad_and_scale
from dx_modelzoo.postprocessing.decode_utils import cxcywh_to_xyxy
from dx_modelzoo.postprocessing.nms import nms_numpy


def _empty_pose_result(num_keypoints=17):
    return (
        np.empty((0, 4), dtype=np.float64),
        np.empty(0, dtype=np.float64),
        np.empty(0, dtype=np.float64),
        np.empty((0, num_keypoints, 3), dtype=np.float64),
    )


@POSTPROCESSING_REGISTRY.register("yolov5_pose_decode")
class YoloV5PoseDecode:
    """YOLOv5 Pose: [batch, N, 57] with obj_conf * cls_conf scoring.

    Format: [cx, cy, w, h, obj_conf, cls_conf, kpt_x1, kpt_y1, kpt_c1, ...].
    """

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
        if raw.shape[0] == 0:
            return _empty_pose_result(self.num_keypoints)

        obj_conf = raw[:, 4]
        mask = obj_conf > self.conf_thres
        raw = raw[mask]
        if len(raw) == 0:
            return _empty_pose_result(self.num_keypoints)

        scores = raw[:, 4] * raw[:, 5]
        mask2 = scores > self.conf_thres
        raw = raw[mask2]
        scores = scores[mask2]
        if len(raw) == 0:
            return _empty_pose_result(self.num_keypoints)

        boxes = cxcywh_to_xyxy(raw[:, :4])
        keep = nms_numpy(boxes, scores, self.iou_thres)
        boxes = boxes[keep]
        scores = scores[keep]

        kpt_data = raw[keep, 6:]
        n_kpts = kpt_data.shape[1] // 3
        keypoints = kpt_data.reshape(-1, n_kpts, 3)
        cls_ids = np.zeros(len(boxes), dtype=np.float64)

        result = (boxes, scores, cls_ids, keypoints)
        return self._rescale(result, **kwargs)

    def _rescale(self, result, **kwargs):
        origin_hw = kwargs.get("origin_hw")
        input_hw = kwargs.get("input_hw")
        if origin_hw is None or input_hw is None:
            return result
        boxes, scores, cls_ids, keypoints = result
        if len(boxes) == 0:
            return result
        boxes = boxes.copy()
        boxes[:, :4] = unpad_and_scale(boxes[:, :4], input_hw, origin_hw, pad_resize=self.pad_resize)
        if keypoints is not None and len(keypoints) > 0:
            keypoints = keypoints.copy()
            kp_boxes = keypoints[:, :, :2].reshape(-1, 2)
            pseudo = np.column_stack([kp_boxes, kp_boxes])
            scaled = unpad_and_scale(pseudo, input_hw, origin_hw, pad_resize=self.pad_resize)
            keypoints[:, :, :2] = scaled[:, :2].reshape(keypoints.shape[0], -1, 2)
        return boxes, scores, cls_ids, keypoints
