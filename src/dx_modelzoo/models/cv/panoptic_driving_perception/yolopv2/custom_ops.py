"""YOLOPv2 panoptic driving perception decoding.

YOLOPv2 model outputs (yolopv2_384x640.onnx / .dxnn):
    det0      [1, 255, 48, 80]  — raw YOLO detection head (3 anchors × 85)
    det1      [1, 255, 24, 40]  — raw YOLO detection head
    det2      [1, 255, 12, 20]  — raw YOLO detection head
    output_4  [1, 2, H, W]       — drivable area segmentation
    output_5  [1, 1, H, W]       — lane line segmentation

Neither the ONNX nor the DXNN export carries anchor tensors (older ONNX exports
exposed them as ``output_1/2/3``), so the decoder always uses YOLOPv2's built-in
anchors (see ``YOLOPv2PanopticDecode.DEFAULT_ANCHORS``).

This decoder simultaneously decodes vehicle detections (decode + NMS),
the drivable area segmentation (argmax), and the lane line segmentation
(threshold), cropping the letterbox padding and rescaling to the original
image resolution.
"""

from __future__ import annotations

from typing import List

import numpy as np

from dx_modelzoo.postprocessing import POSTPROCESSING_REGISTRY
from dx_modelzoo.postprocessing.coord_scaler import unpad_and_scale


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> List[int]:
    """Pure-numpy class-agnostic NMS. ``boxes`` in xyxy."""
    if boxes.shape[0] == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thres]
    return keep


def _letterbox_content(model_hw, orig_hw):
    """Return (top, left, content_h, content_w) of the unpadded content region.

    Mirrors YOLOv5/coord_scaler letterbox geometry (symmetric padding).
    """
    h_model, w_model = model_hw
    h_orig, w_orig = orig_hw
    ratio = min(h_model / h_orig, w_model / w_orig)
    content_h = int(round(h_orig * ratio))
    content_w = int(round(w_orig * ratio))
    pad_h = (h_model - content_h) / 2.0
    pad_w = (w_model - content_w) / 2.0
    top = int(round(pad_h - 0.1))
    left = int(round(pad_w - 0.1))
    top = max(0, min(top, h_model - content_h))
    left = max(0, min(left, w_model - content_w))
    return top, left, content_h, content_w


@POSTPROCESSING_REGISTRY.register("yolopv2_panoptic_decode")
class YOLOPv2PanopticDecode:
    """Decode YOLOPv2 multi-task output into detection + drivable + lane.

    Returns a dict with:
      * ``boxes``: ``[N, 5]`` vehicle detections ``(x1, y1, x2, y2, score)`` in
        original image coordinates.
      * ``drivable``: ``[Hc, Wc]`` argmax segmentation map cropped to the
        unpadded model content region (``{0, 1}``).
      * ``lane``: ``[Hc, Wc]`` binary lane map cropped to the same region.

    The drivable/lane maps are returned at model content resolution (no
    upsampling); the evaluator resizes the ground truth to match.

    Args:
        vehicle_class_index: Class channel treated as the single "vehicle"
            class for detection (YOLOPv2 only activates index 3).
        conf_thres: Confidence threshold for detection candidates.
        iou_thres: IoU threshold for NMS.
        max_det: Maximum detections kept per image after NMS.
        num_anchors: Anchors per detection cell (3 for YOLOPv2).
        input_size: Model input spatial size. Either an ``int`` (square) or a
            ``[H, W]`` pair (e.g. ``[384, 640]``). Only used as a fallback when
            the evaluator does not supply ``input_hw``.
        lane_thres: Threshold applied to the lane probability map.
        anchors: Per-scale anchor sizes (pixels) ordered from smallest stride
            (largest grid) to largest stride, e.g.
            ``[[[12, 16], [19, 36], [40, 28]], ...]``. These fixed anchors are
            always used: neither the ONNX nor the DXNN export carries anchor
            tensors, so the decoder relies solely on these defaults.
    """

    DEFAULT_ANCHORS = [
        [[12.0, 16.0], [19.0, 36.0], [40.0, 28.0]],
        [[36.0, 75.0], [76.0, 55.0], [72.0, 146.0]],
        [[142.0, 110.0], [192.0, 243.0], [459.0, 401.0]],
    ]

    def __init__(
        self,
        vehicle_class_index: int = 3,
        conf_thres: float = 0.001,
        iou_thres: float = 0.6,
        max_det: int = 300,
        num_anchors: int = 3,
        input_size=640,
        lane_thres: float = 0.5,
        anchors=None,
        **kwargs,
    ):
        self.vehicle_class_index = vehicle_class_index
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.num_anchors = num_anchors
        self.input_size = input_size
        self.lane_thres = lane_thres
        anchors = anchors if anchors is not None else self.DEFAULT_ANCHORS
        # Anchors are ordered smallest-stride (largest grid) first, matching
        # det_heads sorted by grid size descending. The DXNN export strips the
        # model's anchor outputs, so we always rely on these fixed anchors.
        self.anchor_grids = [np.asarray(scale, dtype=np.float32).reshape(1, num_anchors, 1, 1, 2) for scale in anchors]

    def _fallback_model_hw(self):
        if isinstance(self.input_size, (list, tuple)):
            return int(self.input_size[0]), int(self.input_size[1])
        return int(self.input_size), int(self.input_size)

    def _classify_outputs(self, outputs):
        det_heads, drivable, lane = [], None, None
        for o in outputs:
            arr = np.asarray(o)
            if arr.ndim == 4 and arr.shape[2] > 4 and arr.shape[3] > 4:
                ch = arr.shape[1]
                if ch == 1:
                    lane = arr
                elif ch == 2:
                    drivable = arr
                elif ch % self.num_anchors == 0 and ch >= self.num_anchors * 6:
                    det_heads.append(arr)
        return det_heads, drivable, lane

    def _decode_detection(self, det_heads, model_hw, orig_hw) -> np.ndarray:
        if not det_heads or len(det_heads) != len(self.anchor_grids):
            return np.zeros((0, 5), dtype=np.float32)

        # Pair smallest-stride head (largest grid) with smallest anchors.
        det_heads = sorted(det_heads, key=lambda a: a.shape[2], reverse=True)
        anchor_grids = self.anchor_grids

        cls_idx = self.vehicle_class_index
        model_h = model_hw[0]
        boxes_all, scores_all = [], []
        for det, anch in zip(det_heads, anchor_grids):
            _, ch, ny, nx = det.shape
            na = self.num_anchors
            no = ch // na
            stride = model_h / ny
            d = det.reshape(1, na, no, ny, nx).transpose(0, 1, 3, 4, 2)
            d = _sigmoid(d)
            yv, xv = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
            grid = np.stack([xv, yv], -1).reshape(1, 1, ny, nx, 2)
            xy = (d[..., 0:2] * 2 - 0.5 + grid) * stride
            wh = (d[..., 2:4] * 2) ** 2 * anch.reshape(1, na, 1, 1, 2)
            if no - 5 > cls_idx:
                cls_conf = d[..., 5 + cls_idx]
            else:  # fallback: max over classes
                cls_conf = d[..., 5:].max(-1)
            conf = d[..., 4] * cls_conf
            mask = conf > self.conf_thres
            if not mask.any():
                continue
            cx, cy = xy[..., 0][mask], xy[..., 1][mask]
            ww, hh = wh[..., 0][mask], wh[..., 1][mask]
            boxes_all.append(np.stack([cx - ww / 2, cy - hh / 2, cx + ww / 2, cy + hh / 2], axis=1))
            scores_all.append(conf[mask])

        if not boxes_all:
            return np.zeros((0, 5), dtype=np.float32)

        boxes = np.concatenate(boxes_all, axis=0)
        scores = np.concatenate(scores_all, axis=0)
        keep = _nms(boxes, scores, self.iou_thres)
        keep = keep[: self.max_det]
        boxes, scores = boxes[keep], scores[keep]

        boxes = unpad_and_scale(boxes, model_hw, orig_hw, pad_resize=True)
        return np.concatenate([boxes, scores[:, None]], axis=1).astype(np.float32)

    def _decode_mask(self, arr, model_hw, orig_hw, is_lane: bool) -> np.ndarray:
        a = np.asarray(arr)
        if a.ndim == 4:
            a = a[0]
        if is_lane:
            seg = (a[0] > self.lane_thres).astype(np.int64)
        else:
            seg = np.argmax(a, axis=0).astype(np.int64)
        top, left, ch, cw = _letterbox_content((seg.shape[0], seg.shape[1]), orig_hw)
        return seg[top : top + ch, left : left + cw]

    def __call__(self, outputs, origin_hw=None, input_hw=None, **kwargs):
        det_heads, drivable, lane = self._classify_outputs(outputs)

        if origin_hw is None:
            origin_hw = (720, 1280)
        model_hw = tuple(input_hw) if input_hw is not None else self._fallback_model_hw()
        orig_hw = (int(origin_hw[0]), int(origin_hw[1]))

        boxes = self._decode_detection(det_heads, model_hw, orig_hw)
        drivable_map = (
            self._decode_mask(drivable, model_hw, orig_hw, is_lane=False)
            if drivable is not None
            else np.zeros((1, 1), dtype=np.int64)
        )
        lane_map = (
            self._decode_mask(lane, model_hw, orig_hw, is_lane=True)
            if lane is not None
            else np.zeros((1, 1), dtype=np.int64)
        )
        return {"boxes": boxes, "drivable": drivable_map, "lane": lane_map}
