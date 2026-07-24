"""YOLOv6 output decoding: [1, N, 5+C] obj_conf × cls_scores → NMS input dict."""
from __future__ import annotations

from dx_modelzoo.postprocessing import POSTPROCESSING_REGISTRY
from dx_modelzoo.postprocessing.decode_utils import apply_obj_cls_score, build_nms_input


@POSTPROCESSING_REGISTRY.register("yolov6_decode")
class YOLOv6Decode:
    def __call__(self, outputs, **kwargs):
        out = outputs[0] if isinstance(outputs, list) else outputs
        if out.ndim == 2:
            out = out[None, ...]
        out = out[0]  # [N, 5+C]
        boxes, scores, class_ids = apply_obj_cls_score(out)
        return build_nms_input(boxes, scores, class_ids)
