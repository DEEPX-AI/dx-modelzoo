"""NanoDet output decoding variants with integrated NMS."""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from dx_modelzoo.postprocessing import POSTPROCESSING_REGISTRY
from dx_modelzoo.postprocessing.coord_scaler import unpad_and_scale
from dx_modelzoo.postprocessing.decode_utils import generate_grid_center_priors, infer_input_size
from dx_modelzoo.postprocessing.nms import nms_numpy

MAX_WH = 4096
MAX_NMS = 3000


@POSTPROCESSING_REGISTRY.register("nanodet_decode")
class NanoDetDecode:
    """NanoDet GFL-head: distribution-based bbox decoding + class-wise NMS.

    Includes integrated NMS because decoding is tightly coupled with filtering.
    Returns [M, 6] ndarray directly.
    """

    def __init__(
        self,
        num_class: int = 80,
        reg_max: int = 10,
        strides: Optional[List[int]] = None,
        conf_thres: float = 0.001,
        iou_thres: float = 0.7,
        pad_resize: bool = True,
        center_offset: float = 0.5,
        **kwargs,
    ):
        self.num_class = num_class
        self.reg_max = reg_max
        self.pad_resize = pad_resize
        self.strides = strides or [8, 16, 32]
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        # Legacy NanoDet uses cell-center priors ((x + 0.5) * stride); NanoDet-Plus
        # uses top-left grid priors (x * stride). Set center_offset=0.0 for Plus.
        self.center_offset = center_offset

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
        if isinstance(outputs, list):
            if len(outputs) == 0:
                return np.empty((0, 6), dtype=np.float64)
            feats = outputs[0]
        else:
            feats = outputs

        if feats.ndim == 3:
            if feats.shape[0] == 1:
                feats = feats.squeeze(0)
            elif feats.shape[1] == 1:
                feats = feats.squeeze(1)
        if feats.ndim > 2:
            feats = feats.reshape(-1, feats.shape[-1])

        expected_channels = self.num_class + 4 * (self.reg_max + 1)
        if feats.shape[1] != expected_channels:
            return np.empty((0, 6), dtype=np.float64)

        input_size = infer_input_size(feats.shape[0], self.strides)
        center_priors = generate_grid_center_priors(input_size, input_size, self.strides)
        if len(center_priors) != feats.shape[0]:
            return np.empty((0, 6), dtype=np.float64)

        scores_all = feats[:, : self.num_class].astype(np.float32)
        max_scores = np.max(scores_all, axis=1)
        labels = np.argmax(scores_all, axis=1)

        keep_mask = max_scores > self.conf_thres
        if not np.any(keep_mask):
            return np.empty((0, 6), dtype=np.float64)

        sel_idxs = np.nonzero(keep_mask)[0]
        sel_scores = max_scores[sel_idxs].astype(np.float32)
        sel_labels = labels[sel_idxs].astype(np.int64)
        sel_centers = center_priors[sel_idxs].astype(np.float32)

        # Bbox distribution decoding: softmax → expectation
        bbox_preds = feats[sel_idxs, self.num_class :].astype(np.float32)
        n = bbox_preds.shape[0]
        bbox_preds = bbox_preds.reshape(n, 4, self.reg_max + 1)

        x = bbox_preds - np.max(bbox_preds, axis=-1, keepdims=True)
        exp_x = np.exp(x)
        prob = exp_x / (np.sum(exp_x, axis=-1, keepdims=True) + 1e-12)

        coef = np.arange(self.reg_max + 1, dtype=np.float32)
        dis = np.sum(prob * coef.reshape(1, 1, -1), axis=-1)

        strides_sel = sel_centers[:, 2].reshape(-1, 1)
        dis = dis * strides_sel

        ct_x = (sel_centers[:, 0] + self.center_offset) * strides_sel[:, 0]
        ct_y = (sel_centers[:, 1] + self.center_offset) * strides_sel[:, 0]

        x1 = np.clip(ct_x - dis[:, 0], 0.0, float(input_size))
        y1 = np.clip(ct_y - dis[:, 1], 0.0, float(input_size))
        x2 = np.clip(ct_x + dis[:, 2], 0.0, float(input_size))
        y2 = np.clip(ct_y + dis[:, 3], 0.0, float(input_size))

        boxes = np.stack([x1, y1, x2, y2], axis=1)

        if len(boxes) > MAX_NMS:
            top = np.argsort(sel_scores)[-MAX_NMS:]
            boxes, sel_scores, sel_labels = boxes[top], sel_scores[top], sel_labels[top]

        offset_boxes = boxes + (sel_labels[:, None].astype(np.float64) * MAX_WH)
        keep = nms_numpy(offset_boxes, sel_scores, self.iou_thres)

        if len(keep) == 0:
            return np.empty((0, 6), dtype=np.float64)

        result = np.column_stack([boxes[keep], sel_scores[keep], sel_labels[keep].astype(np.float64)])
        return self._rescale_result(result, **kwargs)


@POSTPROCESSING_REGISTRY.register("nanodet_repvgg_decode")
class NanoDetRepVGGDecode:
    """NanoDet-RepVGG: YOLOX-style box decode (obj_conf × cls) + NMS.

    Includes integrated NMS. Returns [M, 6] ndarray directly.
    """

    def __init__(
        self,
        num_class: int = 80,
        conf_thres: float = 0.001,
        iou_thres: float = 0.7,
        pad_resize: bool = True,
        **kwargs,
    ):
        self.num_class = num_class
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.pad_resize = pad_resize

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
        if isinstance(outputs, list):
            if len(outputs) == 0:
                return np.empty((0, 6), dtype=np.float64)
            feats = outputs[0]
        else:
            feats = outputs

        if feats.ndim == 3 and feats.shape[0] == 1:
            feats = feats.squeeze(0)
        if feats.ndim != 2:
            return np.empty((0, 6), dtype=np.float64)

        expected_cols = 4 + 1 + self.num_class
        if feats.shape[1] != expected_cols:
            return np.empty((0, 6), dtype=np.float64)

        strides = [8, 16, 32]
        input_size = infer_input_size(feats.shape[0], strides)
        grid_indices = []
        grid_strides = []
        for stride in strides:
            gs = input_size // stride
            for gy in range(gs):
                for gx in range(gs):
                    grid_indices.append((gx, gy))
                    grid_strides.append(stride)

        grid_indices_arr = np.array(grid_indices, dtype=np.float32)
        grid_strides_arr = np.array(grid_strides, dtype=np.float32)

        box_raw = feats[:, :4]
        obj_conf = feats[:, 4:5]
        class_scores = feats[:, 5:]

        scores = obj_conf * class_scores
        max_scores = np.max(scores, axis=1)
        labels_arr = np.argmax(scores, axis=1)

        keep_mask = max_scores > self.conf_thres
        if not np.any(keep_mask):
            return np.empty((0, 6), dtype=np.float64)

        sel_idxs = np.nonzero(keep_mask)[0]
        sel_scores = max_scores[sel_idxs].astype(np.float32)
        sel_labels = labels_arr[sel_idxs].astype(np.int64)
        sel_box = box_raw[sel_idxs]
        sel_grids = grid_indices_arr[sel_idxs]
        sel_strides = grid_strides_arr[sel_idxs]

        # YOLOX-style decode
        cx = (sel_box[:, 0] + sel_grids[:, 0]) * sel_strides
        cy = (sel_box[:, 1] + sel_grids[:, 1]) * sel_strides
        bw = np.exp(np.clip(sel_box[:, 2], -50.0, 50.0)) * sel_strides
        bh = np.exp(np.clip(sel_box[:, 3], -50.0, 50.0)) * sel_strides

        x1 = np.clip(cx - bw / 2, 0.0, float(input_size))
        y1 = np.clip(cy - bh / 2, 0.0, float(input_size))
        x2 = np.clip(cx + bw / 2, 0.0, float(input_size))
        y2 = np.clip(cy + bh / 2, 0.0, float(input_size))

        boxes = np.stack([x1, y1, x2, y2], axis=1)

        if len(boxes) > MAX_NMS:
            top = np.argsort(sel_scores)[-MAX_NMS:]
            boxes, sel_scores, sel_labels = boxes[top], sel_scores[top], sel_labels[top]

        offset_boxes = boxes + (sel_labels[:, None].astype(np.float64) * MAX_WH)
        keep = nms_numpy(offset_boxes, sel_scores, self.iou_thres)

        if len(keep) == 0:
            return np.empty((0, 6), dtype=np.float64)

        result = np.column_stack([boxes[keep], sel_scores[keep], sel_labels[keep].astype(np.float64)])
        return self._rescale_result(result, **kwargs)
