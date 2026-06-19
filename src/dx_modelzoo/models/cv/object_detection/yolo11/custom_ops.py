"""YOLO11 output decoding: [1, 4+C, N] → transpose → split → NMS input dict."""
from __future__ import annotations

from dx_modelzoo.postprocessing import POSTPROCESSING_REGISTRY
from dx_modelzoo.postprocessing.decode_utils import build_nms_input, split_box_cls, transpose_output


@POSTPROCESSING_REGISTRY.register("yolo11_decode")
class YOLO11Decode:
    def __call__(self, outputs, **kwargs):
        out = outputs[0] if isinstance(outputs, list) else outputs
        out = transpose_output(out)
        out = out[0]
        boxes, scores, class_ids = split_box_cls(out)
        return build_nms_input(boxes, scores, class_ids)
