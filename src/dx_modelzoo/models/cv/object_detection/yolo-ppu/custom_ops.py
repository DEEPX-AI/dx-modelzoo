"""PPU postprocessor for YOLO models running on NPU hardware.

The NPU PPU (Post-Processing Unit) outputs a 32-byte-per-detection format
containing raw box coordinates, grid info, scores, and labels. This module
decodes that format and applies NMS.

For ONNX inference, the standard output format is detected and delegated
to the appropriate NMS variant.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from dx_modelzoo.postprocessing import POSTPROCESSING_REGISTRY
from dx_modelzoo.postprocessing.coord_scaler import unpad_and_scale
from dx_modelzoo.postprocessing.decode_utils import (
    apply_obj_cls_score,
    build_nms_input,
    build_yolox_grids,
    infer_input_size,
    split_box_cls,
    transpose_output,
)
from dx_modelzoo.postprocessing.nms import NMS, nms_numpy


def _decode_ppu_output(
    outputs: np.ndarray,
    anchors: Optional[list] = None,
    strides: Optional[list] = None,
    yolo_version: Optional[str] = None,
    box_format: str = "xyxy",
) -> tuple:
    """Decode 32-byte PPU hardware output into boxes, scores, labels.

    PPU output layout per detection (32 bytes):
      [0:16]  bbox (4 x float32)
      [16:20] grid_info (4 x uint8: gY, gX, anchor_idx, layer_idx)
      [20:24] score (float32)
      [24:28] label (uint32 as float32)
      [28:32] padding

    Returns (boxes_xyxy, scores, labels) as numpy arrays.
    """
    bboxes = outputs[:, :, :16].view(np.float32)
    scores = outputs[:, :, 20:24].view(np.float32)
    labels = outputs[:, :, 24:28].view(np.uint32).astype(np.float32)

    grid_info = outputs[0][:, 16:20].view(np.uint8)
    gY = grid_info[:, 0].astype(np.float32)
    gX = grid_info[:, 1].astype(np.float32)
    anchor_idx = grid_info[:, 2]
    layer_idx = grid_info[:, 3]

    if strides is None:
        strides_arr = np.array([8, 16, 32, 64], dtype=np.float32)
    else:
        strides_arr = np.array(strides, dtype=np.float32)

    boxes = bboxes[0]
    stride = strides_arr[layer_idx]

    if yolo_version == "x":
        # YOLOX (anchor-free)
        boxes_cx = (boxes[:, 0] + gX) * stride
        boxes_cy = (boxes[:, 1] + gY) * stride
        boxes_w = np.exp(boxes[:, 2]) * stride
        boxes_h = np.exp(boxes[:, 3]) * stride
    elif yolo_version in ("v3", "v4", "v5", "v7"):
        if not anchors:
            raise ValueError(f"Anchors required for YOLO {yolo_version}")
        anchors_by_stride = {
            s: np.array(anchors[i], dtype=np.float32) for i, s in enumerate(strides_arr[: len(anchors)])
        }
        anchor_w = np.zeros(len(boxes), dtype=np.float32)
        anchor_h = np.zeros(len(boxes), dtype=np.float32)
        for s, anch in anchors_by_stride.items():
            mask = stride == s
            if np.any(mask):
                anchor_w[mask] = anch[anchor_idx[mask], 0]
                anchor_h[mask] = anch[anchor_idx[mask], 1]

        boxes_cx = (boxes[:, 0] * 2.0 - 0.5 + gX) * stride
        boxes_cy = (boxes[:, 1] * 2.0 - 0.5 + gY) * stride
        boxes_w = (boxes[:, 2] ** 2 * 4.0) * anchor_w
        boxes_h = (boxes[:, 3] ** 2 * 4.0) * anchor_h
    elif yolo_version is None:
        # Pre-decoded boxes — just convert format
        if box_format == "cxcywh":
            cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            boxes_xyxy = np.column_stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        elif box_format == "xywh":
            boxes_xyxy = np.column_stack(
                [
                    boxes[:, 0],
                    boxes[:, 1],
                    boxes[:, 0] + boxes[:, 2],
                    boxes[:, 1] + boxes[:, 3],
                ]
            )
        elif box_format == "yxyx":
            boxes_xyxy = np.column_stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]])
        else:
            boxes_xyxy = boxes[:, :4].copy()
        return boxes_xyxy, scores[0, :, 0], labels[0, :, 0]
    else:
        raise ValueError(f"Unsupported YOLO version: {yolo_version}")

    boxes_xyxy = np.column_stack(
        [
            boxes_cx - boxes_w * 0.5,
            boxes_cy - boxes_h * 0.5,
            boxes_cx + boxes_w * 0.5,
            boxes_cy + boxes_h * 0.5,
        ]
    )
    return boxes_xyxy, scores[0, :, 0], labels[0, :, 0]


@POSTPROCESSING_REGISTRY.register("ppu_nms")
class PPU_NMS:
    """PPU-aware NMS postprocessor.

    Auto-detects PPU hardware format (shape[-1] == 32) and decodes
    using anchors/strides/yolo_version. Falls back to standard NMS
    for ONNX output.
    """

    def __init__(
        self,
        variant: Optional[str] = None,
        anchors: Optional[list] = None,
        yolo_version: Optional[str] = None,
        box_format: str = "xyxy",
        strides: Optional[list] = None,
        conf_thres: float = 0.001,
        iou_thres: float = 0.7,
        max_output_boxes: int = 300,
        max_wh: int = 7680,
        max_nms: int = 30000,
        pad_resize: bool = True,
        **kwargs,
    ) -> None:
        self.variant = variant
        self.anchors = anchors
        self.yolo_version = yolo_version
        self.box_format = box_format
        self.strides = strides
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_output_boxes = max_output_boxes
        self.max_wh = max_wh
        self.max_nms = max_nms
        self.pad_resize = pad_resize

        # Fallback NMS for ONNX inference
        self._fallback_nms = NMS(
            variant=variant,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            max_output_boxes=max_output_boxes,
            **kwargs,
        )

    def _rescale_result(self, result, **kwargs):
        """Rescale box coordinates from model-input space to original image space."""
        origin_hw = kwargs.get("origin_hw")
        input_hw = kwargs.get("input_hw")
        if origin_hw is None or input_hw is None or len(result) == 0:
            return result
        result = result.copy()
        result[:, :4] = unpad_and_scale(
            result[:, :4],
            input_hw,
            origin_hw,
            pad_resize=self.pad_resize,
        )
        return result

    def __call__(self, outputs, runtime_target: Optional[str] = None, **kwargs):
        out = outputs[0] if isinstance(outputs, list) else outputs

        if out.shape[-1] == 0:
            return np.empty((0, 6), dtype=np.float64)

        # PPU hardware format: 32 bytes per detection
        if out.shape[-1] == 32:
            result = self._ppu_decode_and_nms(outputs)
            return self._rescale_result(result, **kwargs)

        decoded = self._decode_onnx_output(out, runtime_target=runtime_target)
        if decoded is not None:
            return self._fallback_nms(decoded, **kwargs)

        # Standard format: delegate to normal NMS
        return self._fallback_nms(outputs, **kwargs)

    def _decode_onnx_output(self, out: np.ndarray, runtime_target: Optional[str] = None):
        if runtime_target != "onnx":
            return None

        if self.yolo_version == "x":
            return self._decode_onnx_yolox(out)

        if self.variant == "yolov10":
            return self._decode_onnx_yolov10(out)

        if self.variant in {"yolov8", "yolov9", "yolo11"}:
            return self._decode_onnx_split_box_cls(out)

        if self.variant == "yolo" or self.yolo_version in {"v3", "v4", "v5", "v7"}:
            return self._decode_onnx_obj_cls(out)

        return None

    def _decode_onnx_split_box_cls(self, out: np.ndarray):
        transposed = transpose_output(out)
        if transposed.ndim == 2:
            transposed = transposed[np.newaxis, ...]
        if transposed.ndim != 3 or transposed.shape[0] == 0:
            return None

        decoded = transposed[0]
        if decoded.shape[-1] < 5:
            return None

        boxes, scores, class_ids = split_box_cls(decoded.astype(np.float32))
        return build_nms_input(boxes, scores, class_ids)

    def _decode_onnx_yolov10(self, out: np.ndarray):
        """YOLOv10 end-to-end: output is already [1, N, 6] (x1,y1,x2,y2,score,class)."""
        if out.ndim == 3:
            out = out[0]
        if out.ndim != 2 or out.shape[-1] < 6:
            return None
        mask = out[:, 4] > self.conf_thres
        out = out[mask]
        if out.shape[0] > self.max_output_boxes:
            out = out[: self.max_output_boxes]
        return out

    def _decode_onnx_obj_cls(self, out: np.ndarray):
        if out.ndim == 2:
            out = out[np.newaxis, ...]
        if out.ndim != 3 or out.shape[0] == 0 or out.shape[-1] < 6:
            return None

        decoded = out[0].astype(np.float32)
        boxes, scores, class_ids = apply_obj_cls_score(decoded)
        return build_nms_input(boxes, scores, class_ids)

    def _decode_onnx_yolox(self, out: np.ndarray):
        if out.ndim == 2:
            out = out[np.newaxis, ...]

        strides = self.strides or [8, 16, 32]
        input_size = infer_input_size(out.shape[1], strides)
        grids, stride_arr = build_yolox_grids(input_size, strides)

        decoded = out.copy().astype(np.float32)
        n = min(decoded.shape[1], grids.shape[1])
        decoded[:, :n, 0:2] = (decoded[:, :n, 0:2] + grids[:, :n]) * stride_arr[:, :n]
        decoded[:, :n, 2:4] = np.exp(decoded[:, :n, 2:4]) * stride_arr[:, :n]

        boxes, scores, class_ids = apply_obj_cls_score(decoded[0])
        return build_nms_input(boxes, scores, class_ids)

    def _ppu_decode_and_nms(self, outputs):
        out = outputs[0] if isinstance(outputs, list) else outputs

        boxes, scores, labels = _decode_ppu_output(
            out,
            anchors=self.anchors,
            strides=self.strides,
            yolo_version=self.yolo_version,
            box_format=self.box_format,
        )

        # Confidence filter
        mask = scores > self.conf_thres
        if not np.any(mask):
            return np.empty((0, 6), dtype=np.float64)

        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]

        # Sort by score descending
        order = np.argsort(scores)[::-1]
        if len(order) > self.max_nms:
            order = order[: self.max_nms]
        boxes = boxes[order]
        scores = scores[order]
        labels = labels[order]

        # Batched NMS (offset by class)
        offset_boxes = boxes + (labels[:, None] * self.max_wh)
        keep = nms_numpy(offset_boxes, scores, self.iou_thres)

        if len(keep) > self.max_output_boxes:
            keep = keep[: self.max_output_boxes]

        result = np.column_stack(
            [
                boxes[keep],
                scores[keep],
                labels[keep],
            ]
        ).astype(np.float64)
        return result
