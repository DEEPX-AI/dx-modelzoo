"""MediaPipe Palm / Hand detector (SSD) postprocessing custom op.

The 192x192 palm-detection model emits, per anchor:
- ``Identity``   [1, 2016, 18]  regressors: 4 bbox (x, y, w, h) + 7 keypoints * 2
- ``Identity_1`` [1, 2016, 1]   score logits

Graph layout (verified from the ONNX): the regressor/classifier heads are
concatenated as ``[palm_8 (stride 8, 24x24x2), palm_16 (stride 16, 12x12x6)]``
with per-cell channel order ``[H, W, anchor]`` — matching ``generate_palm_anchors``.

Decoding follows MediaPipe's ``SsdAnchorsCalculator`` /
``TensorsToDetectionsCalculator``:
- anchors generated with strides ``[8, 16, 16, 16]`` (fixed anchor size = 1.0)
- bbox/keypoint offsets are decoded relative to the anchor centers
- scores are ``sigmoid(clip(logit, -100, 100))``

Input expectation (verified empirically against this export): **RGB, scaled to
[0, 1]** (``x / 255``).  Feeding [-1, 1] yields confident scores but saturated
(garbage) regressors.

Palm-to-hand expansion
----------------------
The palm model localizes the *palm*, which is smaller than and offset from the
full-hand bounding box used as ground truth.  Following MediaPipe's
``DetectionsToRects`` + ``RectTransformation``, the palm box is enlarged by
``box_scale`` and shifted by ``box_shift`` along the wrist -> middle-finger
direction (keypoints 0 and 2) to approximate the full-hand box.

Output boxes are normalized ``[0, 1]`` ``xyxy`` relative to the square model
input, directly comparable to normalized GT boxes.
"""
from __future__ import annotations

import math

import numpy as np

from dx_modelzoo.postprocessing import POSTPROCESSING_REGISTRY
from dx_modelzoo.postprocessing.nms import nms_numpy

# MediaPipe palm-detection (192x192) SSD anchor options.
_STRIDES = (8, 16, 16, 16)
_NUM_LAYERS = 4
_ANCHOR_OFFSET = 0.5
_ANCHORS_PER_CELL = 2  # aspect_ratio 1.0 + interpolated_scale 1.0

# Palm keypoints used to orient the hand box (wrist, middle-finger MCP).
_KP_WRIST = 0
_KP_MIDDLE = 2

_ANCHOR_CACHE: dict = {}


def generate_palm_anchors(input_size: int = 192) -> np.ndarray:
    """Generate MediaPipe SSD anchor centers (cx, cy) in normalized coords.

    Returns ``[num_anchors, 2]`` (2016 for 192x192).  Anchor width/height are
    fixed to 1.0 so only centers are needed for decoding.
    """
    if input_size in _ANCHOR_CACHE:
        return _ANCHOR_CACHE[input_size]

    anchors = []
    layer_id = 0
    while layer_id < _NUM_LAYERS:
        last_same = layer_id
        repeats = 0
        while last_same < _NUM_LAYERS and _STRIDES[last_same] == _STRIDES[layer_id]:
            repeats += _ANCHORS_PER_CELL
            last_same += 1
        stride = _STRIDES[layer_id]
        feature_map = math.ceil(input_size / stride)
        for y in range(feature_map):
            for x in range(feature_map):
                cx = (x + _ANCHOR_OFFSET) / feature_map
                cy = (y + _ANCHOR_OFFSET) / feature_map
                for _ in range(repeats):
                    anchors.append((cx, cy))
        layer_id = last_same

    arr = np.asarray(anchors, dtype=np.float32)
    _ANCHOR_CACHE[input_size] = arr
    return arr


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -100.0, 100.0)))


def _split_outputs(outputs):
    """Return (regressors [N,18], score_logits [N]) regardless of output order."""
    reg, score = None, None
    for out in outputs:
        arr = np.asarray(out)
        last = arr.shape[-1]
        if last == 18:
            reg = arr.reshape(-1, 18)
        elif last == 1:
            score = arr.reshape(-1)
    return reg, score


@POSTPROCESSING_REGISTRY.register("mediapipe_palm_decode")
class MediaPipePalmDecode:
    """Decode MediaPipe palm-detection SSD outputs into full-hand detections."""

    def __init__(
        self,
        input_size: int = 192,
        conf_thres: float = 0.5,
        iou_thres: float = 0.3,
        scale: float = 192.0,
        box_scale: float = 2.0,
        box_shift: float = 0.3,
        **kwargs,
    ) -> None:
        self.input_size = int(input_size)
        self.conf_thres = float(conf_thres)
        self.iou_thres = float(iou_thres)
        self.scale = float(scale)
        self.box_scale = float(box_scale)
        self.box_shift = float(box_shift)

    def __call__(self, outputs, **kwargs):
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]
        reg, score_logits = _split_outputs(outputs)
        if reg is None or score_logits is None:
            return []

        anchors = generate_palm_anchors(self.input_size)
        n = min(len(anchors), reg.shape[0])
        anchors = anchors[:n]
        reg = reg[:n]
        scores = _sigmoid(score_logits[:n])

        # reverse_output_order=true -> (x, y, w, h) at indices 0..3.
        cx = reg[:, 0] / self.scale + anchors[:, 0]
        cy = reg[:, 1] / self.scale + anchors[:, 1]
        w = np.abs(reg[:, 2]) / self.scale
        h = np.abs(reg[:, 3]) / self.scale
        s = np.maximum(w, h)

        # Keypoints 0 (wrist) and 2 (middle-finger MCP) for hand orientation.
        kx0 = reg[:, 4 + 2 * _KP_WRIST] / self.scale + anchors[:, 0]
        ky0 = reg[:, 4 + 2 * _KP_WRIST + 1] / self.scale + anchors[:, 1]
        kx2 = reg[:, 4 + 2 * _KP_MIDDLE] / self.scale + anchors[:, 0]
        ky2 = reg[:, 4 + 2 * _KP_MIDDLE + 1] / self.scale + anchors[:, 1]
        dx = kx2 - kx0
        dy = ky2 - ky0
        norm = np.hypot(dx, dy) + 1e-9
        dx /= norm
        dy /= norm

        # Expand palm box to approximate full-hand box.
        side = s * self.box_scale
        ccx = cx + self.box_shift * side * dx
        ccy = cy + self.box_shift * side * dy
        x1 = ccx - side / 2.0
        y1 = ccy - side / 2.0
        x2 = ccx + side / 2.0
        y2 = ccy + side / 2.0
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        mask = scores > self.conf_thres
        boxes, scores = boxes[mask], scores[mask]
        if len(boxes) == 0:
            return []

        keep = nms_numpy(boxes, scores, self.iou_thres)
        boxes, scores = boxes[keep], scores[keep]
        return [[float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(sc)] for b, sc in zip(boxes, scores)]
