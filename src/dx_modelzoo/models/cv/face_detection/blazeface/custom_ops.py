"""BlazeFace output decoder.

Converts ``selectedBoxes [1, N, 16]`` into a compact ``[N, 5]`` array
with confidence appended.  Coordinates remain in normalised [0, 1] range;
the evaluator is responsible for scaling to original image dimensions.
"""
from __future__ import annotations

import numpy as np

from dx_modelzoo.postprocessing import POSTPROCESSING_REGISTRY


@POSTPROCESSING_REGISTRY.register("blazeface_decode")
class BlazeFaceDecode:
    """Decode BlazeFace selectedBoxes [1, N, 16] → [N, 5].

    The raw tensor stores normalised yxyx coordinates in the first four
    columns.  This decoder reorders them and optionally converts to xywh.

    ``output_format``
        * ``"xywh"`` – ``[x1, y1, w, h, score]``  (default, WiderFace style)
        * ``"xyxy"`` – ``[x1, y1, x2, y2, score]``

    All boxes that survived the on-device NMS are assigned
    ``score = 1.0``.
    """

    def __init__(self, output_format: str = "xywh", **kwargs):
        if output_format not in ("xywh", "xyxy"):
            raise ValueError(f"output_format must be 'xywh' or 'xyxy', got '{output_format}'")
        self.output_format = output_format

    def __call__(self, outputs, **kwargs):
        raw = np.asarray(outputs[0] if isinstance(outputs, (list, tuple)) else outputs)
        if raw.ndim == 3:
            raw = raw[0]  # [N, 16]

        if raw.shape[0] == 0 or raw.shape[1] < 4:
            return np.empty((0, 5), dtype=np.float32)

        # Normalised yxyx → individual coordinates
        y1, x1, y2, x2 = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]
        score = np.ones(raw.shape[0], dtype=np.float32)

        if self.output_format == "xywh":
            return np.stack([x1, y1, x2 - x1, y2 - y1, score], axis=1)
        # xyxy
        return np.stack([x1, y1, x2, y2, score], axis=1)
