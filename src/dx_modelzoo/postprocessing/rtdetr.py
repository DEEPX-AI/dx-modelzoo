from __future__ import annotations

import numpy as np

from dx_modelzoo.postprocessing import POSTPROCESSING_REGISTRY


def unpad_and_scale(boxes: np.ndarray, input_hw, origin_hw, pad_location: str) -> np.ndarray:
    if input_hw is None or origin_hw is None:
        return boxes
    h_model, w_model = (float(v) for v in input_hw)
    h_orig, w_orig = (float(v) for v in origin_hw)
    ratio = min(h_model / h_orig, w_model / w_orig)
    result = boxes.copy()
    if pad_location == "direct":
        result[:, [0, 2]] *= w_orig / w_model
        result[:, [1, 3]] *= h_orig / h_model
    else:
        pad_w = (w_model - w_orig * ratio) / 2 if pad_location == "edge" else 0.0
        pad_h = (h_model - h_orig * ratio) / 2 if pad_location == "edge" else 0.0
        result[:, [0, 2]] = (result[:, [0, 2]] - pad_w) / ratio
        result[:, [1, 3]] = (result[:, [1, 3]] - pad_h) / ratio
    result[:, [0, 2]] = np.clip(result[:, [0, 2]], 0, w_orig)
    result[:, [1, 3]] = np.clip(result[:, [1, 3]], 0, h_orig)
    return result


def decode_rows(outputs, input_hw=None, origin_hw=None, pad_location="edge"):
    detections = np.asarray(outputs[0] if isinstance(outputs, (list, tuple)) else outputs)
    if isinstance(outputs, (list, tuple)) and len(outputs) > 1:
        count = int(np.asarray(outputs[1]).reshape(-1)[0])
        detections = detections[:count]
    if detections.ndim == 3:
        detections = detections[0]
    if detections.size == 0:
        return np.empty((0, 6), dtype=np.float64)
    boxes = unpad_and_scale(detections[:, 2:6].astype(np.float64), input_hw, origin_hw, pad_location)
    return np.column_stack([boxes, detections[:, 1], detections[:, 0]]).astype(np.float64)


@POSTPROCESSING_REGISTRY.register("rtdetr_decode")
class RTDETRDecode:
    """Decode PaddleDetection RT-DETR ``[detections, count]`` output."""

    def __init__(self, pad_location: str = "edge", **kwargs) -> None:
        self.pad_location = pad_location

    def __call__(self, outputs, **kwargs):
        return decode_rows(
            outputs,
            input_hw=kwargs.get("input_hw"),
            origin_hw=kwargs.get("origin_hw"),
            pad_location=self.pad_location,
        )
