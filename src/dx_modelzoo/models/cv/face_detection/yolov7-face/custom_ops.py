"""YOLOv7 face detection postprocessing custom ops."""
from __future__ import annotations

from dx_modelzoo.postprocessing import POSTPROCESSING_REGISTRY
from dx_modelzoo.postprocessing.coord_scaler import unpad_and_scale, xyxy_to_xywh_list
from dx_modelzoo.postprocessing.decode_utils import cxcywh_to_xyxy
from dx_modelzoo.postprocessing.nms import nms_numpy


@POSTPROCESSING_REGISTRY.register("yolov7_face_decode")
class YoloV7FaceDecode:
    """YOLOv7 face detection decoder.

    Output format differs from standard YOLO (landmarks between box and class):
    - YOLOv7 face: [cx, cy, w, h, obj_conf, cls_conf, lm1_x..lm5_y]
    - YOLOv5 face: [cx, cy, w, h, obj_conf, lm1_x..lm5_y, cls_conf]
    """

    def __init__(
        self,
        variant: str = "yolov7_face",
        conf_thres: float = 0.02,
        iou_thres: float = 0.5,
        **kwargs,
    ) -> None:
        self.variant = variant
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def __call__(self, outputs, origin_hw=None, input_hw=None, **kwargs):
        if input_hw is None:
            input_hw = origin_hw
        if origin_hw is None:
            origin_hw = input_hw

        raw = outputs[0] if isinstance(outputs, list) else outputs
        if raw.ndim == 3:
            raw = raw[0]

        mask = raw[:, 4] > self.conf_thres
        raw = raw[mask]
        if len(raw) == 0:
            return []

        if self.variant == "yolov5_face":
            confidence = raw[:, 4] * raw[:, -1]
        else:
            confidence = raw[:, 4] * raw[:, 5]

        mask2 = confidence > self.conf_thres
        raw = raw[mask2]
        confidence = confidence[mask2]
        if len(raw) == 0:
            return []

        boxes = cxcywh_to_xyxy(raw[:, :4].copy())
        keep = nms_numpy(boxes, confidence, self.iou_thres)
        boxes = boxes[keep]
        scores = confidence[keep]

        boxes = unpad_and_scale(boxes, input_hw, origin_hw, pad_resize=True)
        return xyxy_to_xywh_list(boxes, scores)
