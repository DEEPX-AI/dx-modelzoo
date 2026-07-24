"""YOLOX output decoding: grid-based decode + obj_conf × cls."""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from dx_modelzoo.postprocessing import POSTPROCESSING_REGISTRY
from dx_modelzoo.postprocessing.decode_utils import apply_obj_cls_score, build_nms_input, build_yolox_grids


@POSTPROCESSING_REGISTRY.register("yolox_decode")
class YOLOXDecode:
    """YOLOX: grid-based xy/wh decode → obj_conf × cls → NMS input."""

    def __init__(
        self,
        input_size: int = 640,
        strides: Optional[List[int]] = None,
        **kwargs,
    ):
        self.input_size = input_size
        self.strides = strides or [8, 16, 32]
        self._grids, self._strides = build_yolox_grids(input_size, self.strides)

    def __call__(self, outputs, **kwargs):
        out = outputs[0] if isinstance(outputs, list) else outputs
        if out.ndim == 2:
            out = out[np.newaxis, ...]
        out = out.copy().astype(np.float32)

        n = min(out.shape[1], self._grids.shape[1])
        out[:, :n, 0:2] = (out[:, :n, 0:2] + self._grids[:, :n]) * self._strides[:, :n]
        out[:, :n, 2:4] = np.exp(out[:, :n, 2:4]) * self._strides[:, :n]

        decoded = out[0]  # [N, 5+C]
        boxes, scores, class_ids = apply_obj_cls_score(decoded)
        return build_nms_input(boxes, scores, class_ids)
