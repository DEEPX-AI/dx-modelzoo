"""CenterPose postprocessing custom ops."""
from __future__ import annotations

import numpy as np

from dx_modelzoo.postprocessing import POSTPROCESSING_REGISTRY
from dx_modelzoo.postprocessing.coord_scaler import unpad_and_scale


def _nms_heatmap(heatmap: np.ndarray, kernel: int = 3) -> np.ndarray:
    """Max-pooling based NMS on heatmap to find local peaks."""
    pad = kernel // 2
    padded = np.pad(heatmap, pad, mode="constant", constant_values=0)
    h, w = heatmap.shape
    maxpool = np.zeros_like(heatmap)
    for i in range(h):
        for j in range(w):
            maxpool[i, j] = padded[i : i + kernel, j : j + kernel].max()
    return heatmap * (heatmap == maxpool).astype(heatmap.dtype)


def _empty_pose_result(num_keypoints=17):
    return (
        np.empty((0, 4), dtype=np.float64),
        np.empty(0, dtype=np.float64),
        np.empty(0, dtype=np.float64),
        np.empty((0, num_keypoints, 3), dtype=np.float64),
    )


@POSTPROCESSING_REGISTRY.register("centerpose_decode")
class CenterPoseDecode:
    """CenterPose: heatmap-based detection with 6 output heads."""

    def __init__(
        self,
        conf_thres: float = 0.25,
        iou_thres: float = 0.7,
        num_keypoints: int = 17,
        input_size: int = 512,
        pad_resize: bool = True,
        **kwargs,
    ) -> None:
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.num_keypoints = num_keypoints
        self.input_size = input_size
        self.pad_resize = pad_resize

    def __call__(self, outputs, **kwargs):
        result = self._decode(outputs)
        return self._rescale(result, **kwargs)

    def _decode(self, outputs):
        if len(outputs) < 6:
            return _empty_pose_result(self.num_keypoints)

        hm = outputs[0][0, 0]
        wh = outputs[1][0]
        hp = outputs[2][0]
        reg = outputs[3][0]
        hps = outputs[4][0]
        hp_offset = outputs[5][0]

        H, W = hm.shape
        stride = self.input_size // H

        hm = 1.0 / (1.0 + np.exp(-hm.clip(-50, 50)))
        hm = _nms_heatmap(hm)

        flat = hm.ravel()
        if len(flat) == 0:
            return _empty_pose_result(self.num_keypoints)

        top_k = min(300, len(flat))
        top_inds = np.argpartition(flat, -top_k)[-top_k:]
        top_scores = flat[top_inds]

        mask = top_scores > self.conf_thres
        top_inds = top_inds[mask]
        top_scores = top_scores[mask]

        if len(top_inds) == 0:
            return _empty_pose_result(self.num_keypoints)

        ys = top_inds // W
        xs = top_inds % W

        cx = (xs.astype(np.float32) + reg[0, ys, xs]) * stride
        cy = (ys.astype(np.float32) + reg[1, ys, xs]) * stride

        bw = wh[0, ys, xs] * stride
        bh = wh[1, ys, xs] * stride
        x1 = np.clip(cx - bw / 2, 0, self.input_size)
        y1 = np.clip(cy - bh / 2, 0, self.input_size)
        x2 = np.clip(cx + bw / 2, 0, self.input_size)
        y2 = np.clip(cy + bh / 2, 0, self.input_size)

        nk = self.num_keypoints
        kps = np.zeros((len(top_inds), nk, 3), dtype=np.float32)
        for j in range(nk):
            kp_x = cx + hp[j * 2, ys, xs] * stride
            kp_y = cy + hp[j * 2 + 1, ys, xs] * stride
            kp_hm = 1.0 / (1.0 + np.exp(-hps[j].clip(-50, 50)))
            kp_gx = np.clip(np.round(kp_x / stride).astype(int), 0, W - 1)
            kp_gy = np.clip(np.round(kp_y / stride).astype(int), 0, H - 1)
            kp_x += hp_offset[0, kp_gy, kp_gx] * stride
            kp_y += hp_offset[1, kp_gy, kp_gx] * stride
            kps[:, j, 0] = np.clip(kp_x, 0, self.input_size)
            kps[:, j, 1] = np.clip(kp_y, 0, self.input_size)
            kps[:, j, 2] = kp_hm[kp_gy, kp_gx]

        boxes = np.stack([x1, y1, x2, y2], axis=1)
        cls_ids = np.zeros(len(top_inds), dtype=np.float64)
        return boxes, top_scores, cls_ids, kps

    def _rescale(self, result, **kwargs):
        origin_hw = kwargs.get("origin_hw")
        input_hw = kwargs.get("input_hw")
        if origin_hw is None or input_hw is None:
            return result
        boxes, scores, cls_ids, keypoints = result
        if len(boxes) == 0:
            return result
        boxes = boxes.copy()
        boxes[:, :4] = unpad_and_scale(boxes[:, :4], input_hw, origin_hw, pad_resize=self.pad_resize)
        if keypoints is not None and len(keypoints) > 0:
            keypoints = keypoints.copy()
            kp_boxes = keypoints[:, :, :2].reshape(-1, 2)
            pseudo = np.column_stack([kp_boxes, kp_boxes])
            scaled = unpad_and_scale(pseudo, input_hw, origin_hw, pad_resize=self.pad_resize)
            keypoints[:, :, :2] = scaled[:, :2].reshape(keypoints.shape[0], -1, 2)
        return boxes, scores, cls_ids, keypoints
