"""Pure NMS postprocessor — no model-specific decoding.

This module provides standalone NMS algorithm functions (``nms_numpy``,
``_hard_nms``) and an ``NMS`` class that operates on already-decoded
detection dicts ``{"boxes", "scores", "class_ids"}``.
"""

from __future__ import annotations

import numpy as np

from dx_modelzoo.postprocessing import POSTPROCESSING_REGISTRY
from dx_modelzoo.postprocessing.coord_scaler import unpad_and_scale

__all__ = ["nms_numpy", "NMS"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_WH = 4096  # max box dimension for class-offset trick in batched NMS
MAX_NMS = 3000  # max candidates fed into NMS kernel


# ---------------------------------------------------------------------------
# Pure algorithm helpers (copied verbatim from nms.py)
# ---------------------------------------------------------------------------
def _hard_nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float, candidate_size: int = 200) -> np.ndarray:
    """Hard NMS with candidate size limit (SSD-style). Returns indices of kept boxes."""
    order = np.argsort(-scores, kind="stable")[:candidate_size]
    keep = []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum((x2 - x1) * (y2 - y1), 0.0)
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-7)
        order = rest[iou <= iou_threshold]
    return np.array(keep, dtype=np.int64)


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Pure numpy NMS. Returns indices of kept boxes."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum((x2 - x1) * (y2 - y1), 0.0)
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=np.int64)


# ---------------------------------------------------------------------------
# NMS postprocessor class
# ---------------------------------------------------------------------------
@POSTPROCESSING_REGISTRY.register("nms")
class NMS:
    """Pure NMS postprocessor.

    Expects dict input from decode step::

        {"boxes": [N, 4] xyxy, "scores": [N], "class_ids": [N]}

    Returns: ``np.ndarray`` of shape ``[M, 6]`` —
    ``(x1, y1, x2, y2, score, class_id)``
    """

    def __init__(
        self,
        conf_thres: float = 0.001,
        iou_thres: float = 0.7,
        max_output_boxes: int = 300,
        pad_resize: bool = True,
        **kwargs,
    ) -> None:
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_output_boxes = max_output_boxes
        self.pad_resize = pad_resize

    def __call__(self, inputs, **kwargs):
        # Dict input (new 2-step pipeline path)
        if isinstance(inputs, dict):
            boxes = inputs["boxes"]  # [N, 4] xyxy
            scores = inputs["scores"]  # [N]
            class_ids = inputs["class_ids"]  # [N]
            extra = inputs.get("extra")  # Optional [N, K]
            result = self._run_nms(boxes, scores, class_ids, extra=extra)
            return self._rescale_coords(result, **kwargs)

        # List/tuple input: unwrap first element (model session returns list)
        if isinstance(inputs, (list, tuple)):
            inputs = inputs[0] if len(inputs) == 1 else np.concatenate(inputs, axis=0)

        # Legacy ndarray [B, N, 6+] input (passthrough — already NMS'd)
        if isinstance(inputs, np.ndarray):
            return inputs

        raise TypeError(f"NMS expects dict or ndarray, got {type(inputs)}")

    def _rescale_coords(self, result: np.ndarray, **kwargs) -> np.ndarray:
        """Rescale box coordinates from model-input space to original image space."""
        origin_hw = kwargs.get("origin_hw")
        input_hw = kwargs.get("input_hw")
        if origin_hw is None or input_hw is None or len(result) == 0:
            return result
        result = result.copy()
        result[:, :4] = unpad_and_scale(
            result[:, :4], input_hw, origin_hw, pad_resize=kwargs.get("pad_resize", self.pad_resize)
        )
        return result

    def _run_nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        extra: np.ndarray | None = None,
    ) -> np.ndarray:
        """Run batched (class-aware) NMS on decoded detections."""
        # 1. Confidence filter
        mask = scores > self.conf_thres
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]
        if extra is not None:
            extra = extra[mask]

        n_extra = extra.shape[1] if extra is not None else 0
        if len(scores) == 0:
            return np.empty((0, 6 + n_extra), dtype=np.float32)

        # 2. Limit candidates by top-k scores
        if len(scores) > MAX_NMS:
            top_idx = np.argsort(-scores, kind="stable")[:MAX_NMS]
            boxes = boxes[top_idx]
            scores = scores[top_idx]
            class_ids = class_ids[top_idx]
            if extra is not None:
                extra = extra[top_idx]

        # 3. Batched NMS — offset boxes by class_id * MAX_WH so that
        #    boxes of different classes never overlap in coordinate space.
        offsets = class_ids[:, None].astype(np.float32) * MAX_WH
        boxes_for_nms = boxes.astype(np.float32) + offsets
        keep = nms_numpy(boxes_for_nms, scores, self.iou_thres)

        # 4. Limit to max_output_boxes
        keep = keep[: self.max_output_boxes]

        # 5. Assemble result [M, 6+K]
        result = np.column_stack([boxes[keep], scores[keep], class_ids[keep]])
        if extra is not None:
            result = np.column_stack([result, extra[keep]])
        return result.astype(np.float32)
