from __future__ import annotations

import numpy as np

from dx_modelzoo.postprocessing import POSTPROCESSING_REGISTRY
from dx_modelzoo.postprocessing.rtdetr import unpad_and_scale


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area + areas - inter, 1e-9)


def _nms(rows: np.ndarray, iou_threshold: float, top_k: int) -> np.ndarray:
    keep = []
    for cls_id in np.unique(rows[:, 5].astype(np.int32)):
        indices = np.flatnonzero(rows[:, 5].astype(np.int32) == cls_id)
        indices = indices[np.argsort(rows[indices, 4])[::-1]]
        while len(indices):
            current = indices[0]
            keep.append(current)
            if len(indices) == 1:
                break
            overlaps = _iou(rows[current, :4], rows[indices[1:], :4])
            indices = indices[1:][overlaps <= iou_threshold]
    keep = np.asarray(keep, dtype=np.int64)
    if len(keep) > top_k:
        keep = keep[np.argsort(rows[keep, 4])[::-1][:top_k]]
    return rows[keep]


@POSTPROCESSING_REGISTRY.register("pp_shitu_decode")
class PPShiTuDecode:
    def __init__(
        self,
        strides=(8, 16, 32, 64),
        reg_max: int = 7,
        score_threshold: float = 0.01,
        iou_threshold: float = 0.6,
        top_k: int = 300,
        pad_location: str = "direct",
        **kwargs,
    ) -> None:
        self.strides = tuple(strides)
        self.reg_max = reg_max
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.top_k = top_k
        self.pad_location = pad_location

    def __call__(self, outputs, **kwargs):
        if isinstance(outputs, np.ndarray):
            return outputs
        cls_heads = outputs[: len(self.strides)]
        reg_heads = outputs[len(self.strides) : len(self.strides) * 2]
        rows = []
        for stride, cls_head, reg_head in zip(self.strides, cls_heads, reg_heads):
            cls_head = np.asarray(cls_head)[0]
            reg_head = np.asarray(reg_head)[0]
            height = int(round(np.sqrt(len(cls_head))))
            width = len(cls_head) // height
            if height * width != len(cls_head):
                raise ValueError(f"Unexpected PP-ShiTu head length: {len(cls_head)}")
            scores = cls_head
            if scores.min() < 0 or scores.max() > 1:
                scores = 1.0 / (1.0 + np.exp(-scores))
            class_ids = scores.argmax(axis=1)
            confidences = scores[np.arange(len(scores)), class_ids]
            keep = confidences >= self.score_threshold
            if not np.any(keep):
                continue
            reg = reg_head[keep].reshape(-1, 4, self.reg_max + 1)
            reg = reg - reg.max(axis=-1, keepdims=True)
            reg = np.exp(reg)
            reg /= reg.sum(axis=-1, keepdims=True)
            reg = (reg * np.arange(self.reg_max + 1, dtype=np.float32)).sum(axis=-1)
            indices = np.flatnonzero(keep)
            xs = (indices % width).astype(np.float32)
            ys = (indices // width).astype(np.float32)
            centers = np.column_stack([xs + 0.5, ys + 0.5]) * stride
            boxes = np.column_stack(
                [
                    centers[:, 0] - reg[:, 0] * stride,
                    centers[:, 1] - reg[:, 1] * stride,
                    centers[:, 0] + reg[:, 2] * stride,
                    centers[:, 1] + reg[:, 3] * stride,
                ]
            )
            rows.append(np.column_stack([boxes, confidences[keep], class_ids[keep]]))
        if not rows:
            return np.empty((0, 6), dtype=np.float64)
        result = np.concatenate(rows).astype(np.float64)
        result[:, :4] = unpad_and_scale(
            result[:, :4], kwargs.get("input_hw"), kwargs.get("origin_hw"), self.pad_location
        )
        return _nms(result, self.iou_threshold, self.top_k)
