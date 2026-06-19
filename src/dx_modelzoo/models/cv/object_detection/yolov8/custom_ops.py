"""YOLOv8 output decoding: [1, 4+C, N] → transpose → split → NMS input dict."""
from __future__ import annotations

from dx_modelzoo.postprocessing import POSTPROCESSING_REGISTRY
from dx_modelzoo.postprocessing.decode_utils import build_nms_input, split_box_cls, transpose_output


@POSTPROCESSING_REGISTRY.register("yolov8_decode")
class YOLOv8Decode:
    def __call__(self, outputs, **kwargs):
        out = outputs[0] if isinstance(outputs, list) else outputs
        out = transpose_output(out)  # [1, C, N] → [1, N, C]
        out = out[0]  # remove batch dim → [N, C]
        boxes, scores, class_ids = split_box_cls(out)
        return build_nms_input(boxes, scores, class_ids)
