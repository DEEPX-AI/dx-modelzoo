"""DarkHRNet pose postprocessing custom op."""
from __future__ import annotations

import numpy as np

from dx_modelzoo.postprocessing import POSTPROCESSING_REGISTRY
from dx_modelzoo.postprocessing.coord_scaler import unpad_and_scale


def _nms_heatmap(heatmap: np.ndarray, kernel: int = 3) -> np.ndarray:
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
    """Decode CenterPose-compatible six-head pose output."""

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
        origin_hw = kwargs.get("origin_hw")
        input_hw = kwargs.get("input_hw")
        if origin_hw is None or input_hw is None or len(result[0]) == 0:
            return result
        boxes, scores, cls_ids, keypoints = result
        boxes = boxes.copy()
        boxes[:, :4] = unpad_and_scale(boxes[:, :4], input_hw, origin_hw, pad_resize=self.pad_resize)
        keypoints = keypoints.copy()
        points = keypoints[:, :, :2].reshape(-1, 2)
        pseudo = np.column_stack([points, points])
        scaled = unpad_and_scale(pseudo, input_hw, origin_hw, pad_resize=self.pad_resize)
        keypoints[:, :, :2] = scaled[:, :2].reshape(keypoints.shape[0], -1, 2)
        return boxes, scores, cls_ids, keypoints

    def _decode(self, outputs):
        if len(outputs) < 6:
            return _empty_pose_result(self.num_keypoints)
        hm = outputs[0][0, 0]
        wh, hp, reg, hps, hp_offset = (outputs[i][0] for i in range(1, 6))
        height, width = hm.shape
        stride = self.input_size // height
        hm = 1.0 / (1.0 + np.exp(-hm.clip(-50, 50)))
        hm = _nms_heatmap(hm)
        flat = hm.ravel()
        if len(flat) == 0:
            return _empty_pose_result(self.num_keypoints)
        top_k = min(300, len(flat))
        inds = np.argpartition(flat, -top_k)[-top_k:]
        scores = flat[inds]
        keep = scores > self.conf_thres
        inds, scores = inds[keep], scores[keep]
        if len(inds) == 0:
            return _empty_pose_result(self.num_keypoints)
        ys, xs = inds // width, inds % width
        cx = (xs.astype(np.float32) + reg[0, ys, xs]) * stride
        cy = (ys.astype(np.float32) + reg[1, ys, xs]) * stride
        bw, bh = wh[0, ys, xs] * stride, wh[1, ys, xs] * stride
        boxes = np.stack(
            [
                np.clip(cx - bw / 2, 0, self.input_size),
                np.clip(cy - bh / 2, 0, self.input_size),
                np.clip(cx + bw / 2, 0, self.input_size),
                np.clip(cy + bh / 2, 0, self.input_size),
            ],
            axis=1,
        )
        keypoints = np.zeros((len(inds), self.num_keypoints, 3), dtype=np.float32)
        for j in range(self.num_keypoints):
            kx = cx + hp[j * 2, ys, xs] * stride
            ky = cy + hp[j * 2 + 1, ys, xs] * stride
            kphm = 1.0 / (1.0 + np.exp(-hps[j].clip(-50, 50)))
            gx = np.clip(np.round(kx / stride).astype(int), 0, width - 1)
            gy = np.clip(np.round(ky / stride).astype(int), 0, height - 1)
            kx += hp_offset[0, gy, gx] * stride
            ky += hp_offset[1, gy, gx] * stride
            keypoints[:, j, 0] = np.clip(kx, 0, self.input_size)
            keypoints[:, j, 1] = np.clip(ky, 0, self.input_size)
            keypoints[:, j, 2] = kphm[gy, gx]
        return boxes, scores, np.zeros(len(inds), dtype=np.float64), keypoints


@POSTPROCESSING_REGISTRY.register("dark_hrnet_heatmap_decode")
class DarkHRNetHeatmapDecode:
    def __init__(self, model_w: float = 192.0, model_h: float = 256.0, **kwargs) -> None:
        self.model_w = float(model_w)
        self.model_h = float(model_h)

    def __call__(self, outputs, **kwargs):
        heatmaps = np.asarray(outputs[0] if isinstance(outputs, (list, tuple)) else outputs)
        if heatmaps.ndim == 4:
            heatmaps = heatmaps[0]
        keypoints, height, width = heatmaps.shape
        flat = heatmaps.reshape(keypoints, -1)
        indices = flat.argmax(axis=1)
        scores = flat.max(axis=1).astype(np.float32)
        x = (indices % width).astype(np.float32)
        y = (indices // width).astype(np.float32)
        for i in range(keypoints):
            xi, yi = int(x[i]), int(y[i])
            if 0 < xi < width - 1:
                x[i] += 0.25 * np.sign(heatmaps[i, yi, xi + 1] - heatmaps[i, yi, xi - 1])
            if 0 < yi < height - 1:
                y[i] += 0.25 * np.sign(heatmaps[i, yi + 1, xi] - heatmaps[i, yi - 1, xi])
        return np.column_stack([x * self.model_w / width, y * self.model_h / height]), scores
