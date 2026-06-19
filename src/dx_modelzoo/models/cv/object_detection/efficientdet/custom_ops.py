"""EfficientDet anchor-based box decoding + class-aware NMS."""
from __future__ import annotations

import numpy as np

from dx_modelzoo.postprocessing import POSTPROCESSING_REGISTRY
from dx_modelzoo.postprocessing.coord_scaler import unpad_and_scale
from dx_modelzoo.postprocessing.nms import nms_numpy

MAX_WH = 4096


def _generate_anchors_from_feature_maps(feature_maps: list[np.ndarray], input_size: int) -> np.ndarray:
    scale_pairs = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
    octave_scales = [1.0, 2 ** (1 / 3), 2 ** (2 / 3)]
    anchor_scale = 4.0

    anchors = []
    for feat in feature_maps:
        feat_h, feat_w = feat.shape[-2:]
        stride_y = input_size / feat_h
        stride_x = input_size / feat_w
        for y in range(feat_h):
            cy = (y + 0.5) * stride_y
            for x in range(feat_w):
                cx = (x + 0.5) * stride_x
                for octave_scale in octave_scales:
                    base_h = stride_y * anchor_scale * octave_scale
                    base_w = stride_x * anchor_scale * octave_scale
                    for width_scale, height_scale in scale_pairs:
                        h = base_h * height_scale
                        w = base_w * width_scale
                        anchors.append([cy - h / 2, cx - w / 2, cy + h / 2, cx + w / 2])

    return np.array(anchors, dtype=np.float32)


@POSTPROCESSING_REGISTRY.register("efficientdet_decode")
class EfficientDetDecode:
    """EfficientDet: anchor-based box decode + class-aware NMS.

    Model outputs (last 3):
      regression (1, N, 4): [dy, dx, dh, dw] offsets
      classification (1, N, C): sigmoid scores
      anchors (1, N, 4): [y1, x1, y2, x2]

    Returns: [M, 6] with 1-indexed class IDs (COCO convention).
    """

    def __init__(
        self,
        input_size: int = 640,
        conf_thres: float = 0.001,
        iou_thres: float = 0.7,
        max_output_boxes: int = 300,
        pad_resize: bool = True,
        **kwargs,
    ):
        self.input_size = input_size
        self.pad_resize = pad_resize
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_output_boxes = max_output_boxes

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
            pad_resize=getattr(self, "pad_resize", True),
        )
        return result

    def __call__(self, outputs, **kwargs):
        if outputs[-3].shape[-1] == 4:
            regression, classification = outputs[-3], outputs[-2]
            anchors = outputs[-1]
        else:
            regression, classification = outputs[-2], outputs[-1]
            anchors = _generate_anchors_from_feature_maps(list(outputs[:-2]), self.input_size)[np.newaxis, ...]

        reg = regression[0]
        cls = classification[0]
        anc = anchors[0]

        wa = anc[:, 3] - anc[:, 1]
        ha = anc[:, 2] - anc[:, 0]
        a_cx = (anc[:, 1] + anc[:, 3]) * 0.5
        a_cy = (anc[:, 0] + anc[:, 2]) * 0.5

        pred_cx = reg[:, 1] * wa + a_cx
        pred_cy = reg[:, 0] * ha + a_cy
        pred_w = np.exp(np.clip(reg[:, 3], -10, 10)) * wa
        pred_h = np.exp(np.clip(reg[:, 2], -10, 10)) * ha

        sz = self.input_size
        x1 = np.clip(pred_cx - 0.5 * pred_w, 0, sz)
        y1 = np.clip(pred_cy - 0.5 * pred_h, 0, sz)
        x2 = np.clip(pred_cx + 0.5 * pred_w, 0, sz)
        y2 = np.clip(pred_cy + 0.5 * pred_h, 0, sz)
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        max_scores = cls.max(axis=1)
        class_indices = cls.argmax(axis=1)

        mask = max_scores > self.conf_thres
        if not np.any(mask):
            return np.empty((0, 6), dtype=np.float64)

        boxes = boxes[mask]
        max_scores = max_scores[mask]
        class_indices = class_indices[mask]

        topk_pre = 2000
        if len(max_scores) > topk_pre:
            topk_idx = np.argpartition(max_scores, -topk_pre)[-topk_pre:]
            boxes = boxes[topk_idx]
            max_scores = max_scores[topk_idx]
            class_indices = class_indices[topk_idx]

        offset_boxes = boxes + (class_indices[:, None].astype(np.float32) * MAX_WH)
        keep = nms_numpy(offset_boxes, max_scores, self.iou_thres)

        if len(keep) > self.max_output_boxes:
            keep = keep[: self.max_output_boxes]

        # COCO 1-indexed class IDs
        result = np.column_stack(
            [
                boxes[keep],
                max_scores[keep],
                (class_indices[keep] + 1).astype(np.float64),
            ]
        ).astype(np.float64)
        return self._rescale_result(result, **kwargs)
