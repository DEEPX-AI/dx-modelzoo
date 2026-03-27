"""Ultra-Light-Fast-Generic-Face-Detector (ULFGFD) models.

Supports RFB and Slim backbones at 320 and 640 resolutions.
Boxes are output in normalized (0-1) coordinates as [x1, y1, x2, y2].
"""

import itertools
import math

import numpy as np
import torch
import torchvision
from torchvision.transforms import Compose

from dx_modelzoo.enums import DatasetType, EvaluationType
from dx_modelzoo.models import ModelBase, ModelInfo
from dx_modelzoo.preprocessing.normalize import Normalize
from dx_modelzoo.preprocessing.resize import Resize
from dx_modelzoo.preprocessing.transpose import Transpose

ULFGFD_STEPS = [8, 16, 32, 64]
ULFGFD_MIN_SIZES = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]


def generate_ulfgfd_priors(image_size, steps=ULFGFD_STEPS, min_sizes=ULFGFD_MIN_SIZES):
    """Generate SSD-style prior boxes for ULFGFD.

    Args:
        image_size: (height, width) of the input image.
        steps: stride for each feature map level.
        min_sizes: anchor sizes for each feature map level.

    Returns:
        np.ndarray: prior boxes [N, 4] in (cx, cy, w, h) format, normalized.
    """
    h, w = image_size
    priors = []
    for k, step in enumerate(steps):
        fh = math.ceil(h / step)
        fw = math.ceil(w / step)
        for i, j in itertools.product(range(fh), range(fw)):
            for ms in min_sizes[k]:
                cx = (j + 0.5) * step / w
                cy = (i + 0.5) * step / h
                priors.append([cx, cy, ms / w, ms / h])
    return np.array(priors, dtype=np.float32)


def ulfgfd_postprocessing(
    outputs,
    inp_shape,
    origin_shape,
    session,
    conf_thres=0.5,
    iou_thres=0.3,
):
    """ULFGFD postprocessing for decoded outputs.

    The model outputs scores [1, N, 2] and boxes [1, N, 4] in normalized
    (0-1) coordinates (x1, y1, x2, y2).

    Args:
        outputs: list of numpy arrays [scores, boxes].
        inp_shape: input tensor shape (N, C, H, W).
        origin_shape: original image shape (H, W, C).
        session: inference session.
        conf_thres: confidence threshold.
        iou_thres: IoU threshold for NMS.

    Returns:
        list: detections as [[x, y, w, h, conf], ...].
    """
    if len(inp_shape) == 4 and inp_shape[3] <= 4:
        inp_shape = [inp_shape[0], inp_shape[3], inp_shape[1], inp_shape[2]]

    h_model, w_model = inp_shape[2], inp_shape[3]
    h_orig, w_orig = origin_shape[0], origin_shape[1]

    # Identify outputs by last dimension
    scores_raw, boxes_raw = None, None
    for out in outputs:
        last_dim = out.shape[-1]
        if last_dim == 2:
            scores_raw = out
        elif last_dim == 4:
            boxes_raw = out

    scores_raw = np.squeeze(scores_raw)  # [N, 2]
    boxes_raw = np.squeeze(boxes_raw)  # [N, 4]

    # Face confidence (column 1)
    scores = scores_raw[:, 1]

    # Convert normalized coords to model pixel coords
    boxes = boxes_raw.copy()
    boxes[:, [0, 2]] *= w_model
    boxes[:, [1, 3]] *= h_model

    # Filter by confidence
    mask = scores > conf_thres
    boxes = boxes[mask]
    scores = scores[mask]

    if len(boxes) == 0:
        return []

    # Scale from model input to original image (direct resize, no padding)
    boxes[:, [0, 2]] *= w_orig / w_model
    boxes[:, [1, 3]] *= h_orig / h_model

    # Clip to original image
    boxes[:, 0] = np.clip(boxes[:, 0], 0, w_orig)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h_orig)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w_orig)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h_orig)

    # NMS
    boxes_t = torch.from_numpy(boxes).float()
    scores_t = torch.from_numpy(scores).float()
    keep = torchvision.ops.nms(boxes_t, scores_t, iou_thres)
    keep = keep.numpy()
    boxes = boxes[keep]
    scores = scores[keep]

    # Convert to [x, y, w, h, conf] format
    result = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        result.append([x1, y1, x2 - x1, y2 - y1, scores[i]])

    return result


def ulfgfd_raw_postprocessing(
    outputs,
    inp_shape,
    origin_shape,
    session,
    conf_thres=0.5,
    iou_thres=0.3,
    variances=(0.1, 0.2),
):
    """ULFGFD postprocessing for raw (without_postprocessing) model outputs.

    Decodes SSD-style raw predictions using prior boxes before applying
    the same detection pipeline as ulfgfd_postprocessing.

    Args:
        outputs: list of numpy arrays [scores, boxes_raw].
        inp_shape: input tensor shape (N, C, H, W).
        origin_shape: original image shape (H, W, C).
        session: inference session.
        conf_thres: confidence threshold.
        iou_thres: IoU threshold for NMS.
        variances: SSD decode variance values (center, size).

    Returns:
        list: detections as [[x, y, w, h, conf], ...].
    """
    if len(inp_shape) == 4 and inp_shape[3] <= 4:
        inp_shape = [inp_shape[0], inp_shape[3], inp_shape[1], inp_shape[2]]

    h_model, w_model = inp_shape[2], inp_shape[3]
    h_orig, w_orig = origin_shape[0], origin_shape[1]

    scores_raw, loc_raw = None, None
    for out in outputs:
        last_dim = out.shape[-1]
        if last_dim == 2:
            scores_raw = out
        elif last_dim == 4:
            loc_raw = out

    scores_raw = np.squeeze(scores_raw)  # [N, 2]
    loc_raw = np.squeeze(loc_raw)  # [N, 4]

    # Decode raw predictions with prior boxes
    priors = generate_ulfgfd_priors((h_model, w_model))

    cx = priors[:, 0] + loc_raw[:, 0] * variances[0] * priors[:, 2]
    cy = priors[:, 1] + loc_raw[:, 1] * variances[0] * priors[:, 3]
    w = priors[:, 2] * np.exp(loc_raw[:, 2] * variances[1])
    h = priors[:, 3] * np.exp(loc_raw[:, 3] * variances[1])

    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

    # Convert normalized coords to model pixel coords
    boxes[:, [0, 2]] *= w_model
    boxes[:, [1, 3]] *= h_model

    scores = scores_raw[:, 1]

    mask = scores > conf_thres
    boxes = boxes[mask]
    scores = scores[mask]

    if len(boxes) == 0:
        return []

    # Scale from model input to original image (direct resize, no padding)
    boxes[:, [0, 2]] *= w_orig / w_model
    boxes[:, [1, 3]] *= h_orig / h_model

    boxes[:, 0] = np.clip(boxes[:, 0], 0, w_orig)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h_orig)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w_orig)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h_orig)

    boxes_t = torch.from_numpy(boxes).float()
    scores_t = torch.from_numpy(scores).float()
    keep = torchvision.ops.nms(boxes_t, scores_t, iou_thres)
    keep = keep.numpy()
    boxes = boxes[keep]
    scores = scores[keep]

    result = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        result.append([x1, y1, x2 - x1, y2 - y1, scores[i]])

    return result


# ---------- RFB backbone ----------


class ULFGFD_RFB_320(ModelBase):
    """ULFGFD with RFB backbone, 320x240 input."""

    info = ModelInfo(
        name="ULFGFD_RFB_320",
        dataset=DatasetType.widerface,
        evaluation=EvaluationType.widerface,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="default", size=(240, 320)),
                Normalize(mean=[127.0, 127.0, 127.0], std=[128.0, 128.0, 128.0]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="default", size=(240, 320)),
            ]
        )

    def postprocessing(self):
        return ulfgfd_postprocessing


class ULFGFD_RFB_320_WO_PP(ModelBase):
    """ULFGFD with RFB backbone, 320x240 input, without postprocessing layer."""

    info = ModelInfo(
        name="ULFGFD_RFB_320_WO_PP",
        dataset=DatasetType.widerface,
        evaluation=EvaluationType.widerface,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="default", size=(240, 320)),
                Normalize(mean=[127.0, 127.0, 127.0], std=[128.0, 128.0, 128.0]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="default", size=(240, 320)),
            ]
        )

    def postprocessing(self):
        return ulfgfd_raw_postprocessing


class ULFGFD_RFB_640(ModelBase):
    """ULFGFD with RFB backbone, 640x480 input."""

    info = ModelInfo(
        name="ULFGFD_RFB_640",
        dataset=DatasetType.widerface,
        evaluation=EvaluationType.widerface,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="default", size=(480, 640)),
                Normalize(mean=[127.0, 127.0, 127.0], std=[128.0, 128.0, 128.0]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="default", size=(480, 640)),
            ]
        )

    def postprocessing(self):
        return ulfgfd_postprocessing


# ---------- Slim backbone ----------


class ULFGFD_Slim_320(ModelBase):
    """ULFGFD with Slim backbone, 320x240 input."""

    info = ModelInfo(
        name="ULFGFD_Slim_320",
        dataset=DatasetType.widerface,
        evaluation=EvaluationType.widerface,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="default", size=(240, 320)),
                Normalize(mean=[127.0, 127.0, 127.0], std=[128.0, 128.0, 128.0]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="default", size=(240, 320)),
            ]
        )

    def postprocessing(self):
        return ulfgfd_postprocessing


class ULFGFD_Slim_320_WO_PP(ModelBase):
    """ULFGFD with Slim backbone, 320x240 input, without postprocessing layer."""

    info = ModelInfo(
        name="ULFGFD_Slim_320_WO_PP",
        dataset=DatasetType.widerface,
        evaluation=EvaluationType.widerface,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="default", size=(240, 320)),
                Normalize(mean=[127.0, 127.0, 127.0], std=[128.0, 128.0, 128.0]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="default", size=(240, 320)),
            ]
        )

    def postprocessing(self):
        return ulfgfd_raw_postprocessing
