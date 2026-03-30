import itertools

import numpy as np
import torch
import torchvision
from torchvision.transforms import Compose

from dx_modelzoo.enums import DatasetType, EvaluationType, SessionType
from dx_modelzoo.models import ModelBase, ModelInfo
from dx_modelzoo.preprocessing.normalize import Normalize
from dx_modelzoo.preprocessing.resize import Resize
from dx_modelzoo.preprocessing.transpose import Transpose


def generate_prior_boxes(
    image_size,
    steps=(8, 16, 32),
    min_sizes=((16, 32), (64, 128), (256, 512)),
):
    """Generate prior boxes (anchors) for RetinaFace.

    Args:
        image_size: (height, width) of the input image.
        steps: stride for each feature map level.
        min_sizes: anchor sizes for each feature map level.

    Returns:
        np.ndarray: prior boxes of shape [N, 4] in (cx, cy, w, h) format, normalized.
    """
    h, w = image_size
    feature_maps = [[h // step, w // step] for step in steps]
    anchors = []
    for k, f in enumerate(feature_maps):
        for i, j in itertools.product(range(f[0]), range(f[1])):
            for min_size in min_sizes[k]:
                cx = (j + 0.5) * steps[k] / w
                cy = (i + 0.5) * steps[k] / h
                s_w = min_size / w
                s_h = min_size / h
                anchors.append([cx, cy, s_w, s_h])
    return np.array(anchors, dtype=np.float32)


def decode_boxes(loc, priors, variances=(0.1, 0.2)):
    """Decode bounding boxes from predictions using prior boxes.

    Args:
        loc: predicted locations [N, 4].
        priors: prior boxes [N, 4] in (cx, cy, w, h) format.
        variances: variance values for decoding.

    Returns:
        np.ndarray: decoded boxes [N, 4] in (x1, y1, x2, y2) format.
    """
    boxes = np.concatenate(
        [
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * np.exp(loc[:, 2:] * variances[1]),
        ],
        axis=1,
    )
    # Convert from cx, cy, w, h to x1, y1, x2, y2
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def retinaface_postprocessing(
    outputs,
    inp_shape,
    origin_shape,
    session,
    conf_thres=0.02,
    iou_thres=0.4,
):
    """RetinaFace postprocessing.

    Args:
        outputs: list of numpy arrays from model inference.
        inp_shape: input tensor shape (N, C, H, W) or (N, H, W, C).
        origin_shape: original image shape (H, W, C).
        session: inference session.
        conf_thres: confidence threshold.
        iou_thres: IoU threshold for NMS.

    Returns:
        list: detections as [[x, y, w, h, conf], ...].
    """
    if session.type == SessionType.dxruntime:
        inp_shape = [inp_shape[0], inp_shape[3], inp_shape[1], inp_shape[2]]

    h_model, w_model = inp_shape[2], inp_shape[3]
    h_orig, w_orig = origin_shape[0], origin_shape[1]

    # Identify outputs by last dimension
    loc, conf, _ = None, None, None  # landms
    for out in outputs:
        last_dim = out.shape[-1]
        if last_dim == 4:
            loc = out
        elif last_dim == 2:
            conf = out
        elif last_dim == 10:
            # landms = out
            pass

    loc = np.squeeze(loc)  # [16800, 4]
    conf = np.squeeze(conf)  # [16800, 2]

    # Generate prior boxes
    priors = generate_prior_boxes((h_model, w_model))

    # Decode boxes to pixel coordinates
    boxes = decode_boxes(loc, priors)
    boxes *= np.array([w_model, h_model, w_model, h_model])

    # Face confidence (column 1)
    scores = conf[:, 1]

    # Filter by confidence
    mask = scores > conf_thres
    boxes = boxes[mask]
    scores = scores[mask]

    if len(boxes) == 0:
        return []

    # Scale from model input to original image (pad resize)
    ratio = min(h_model / h_orig, w_model / w_orig)
    pad_w = (w_model - w_orig * ratio) / 2
    pad_h = (h_model - h_orig * ratio) / 2

    boxes[:, [0, 2]] -= pad_w
    boxes[:, [1, 3]] -= pad_h
    boxes[:, :4] /= ratio

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

    # Convert to [x, y, w, h, conf] format for WiderFace evaluator
    result = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        result.append([x1, y1, x2 - x1, y2 - y1, scores[i]])

    return result


class RetinaFace_MobileNet025(ModelBase):
    """RetinaFace with MobileNet0.25 backbone (folder: retinaface_mobilenet0.25_640-1)."""

    info = ModelInfo(
        name="RetinaFace_MobileNet025",
        dataset=DatasetType.widerface,
        evaluation=EvaluationType.widerface,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=640, pad_location="edge", pad_value=[0, 0, 0]),
                Normalize(mean=[104.0, 117.0, 123.0], std=[1.0, 1.0, 1.0]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=640, pad_location="edge", pad_value=[0, 0, 0]),
            ]
        )

    def postprocessing(self):
        return retinaface_postprocessing


def retinaface_v1_postprocessing(
    outputs,
    inp_shape,
    origin_shape,
    session,
    conf_thres=0.02,
    iou_thres=0.4,
    n_anchors=2,
):
    """RetinaFace postprocessing for per-level spatial outputs.

    Handles models that output separate tensors per FPN level in
    [1, H, W, anchors*dim] format (bbox dim=4, class dim=2, landmark dim=10).

    Args:
        outputs: list of numpy arrays from model inference.
        inp_shape: input tensor shape (N, C, H, W) or (N, H, W, C).
        origin_shape: original image shape (H, W, C).
        session: inference session.
        conf_thres: confidence threshold.
        iou_thres: IoU threshold for NMS.
        n_anchors: number of anchors per spatial location.

    Returns:
        list: detections as [[x, y, w, h, conf], ...].
    """
    if len(inp_shape) == 4 and inp_shape[3] <= 4:
        inp_shape = [inp_shape[0], inp_shape[3], inp_shape[1], inp_shape[2]]

    h_model, w_model = inp_shape[2], inp_shape[3]
    h_orig, w_orig = origin_shape[0], origin_shape[1]

    # Collect and reshape per-level outputs into [N, dim] tensors.
    # Per-level last_dim = n_anchors * per_anchor_dim (e.g. 8=2*4, 4=2*2, 20=2*10).
    loc_list, conf_list = [], []
    for out in outputs:
        out = np.squeeze(out, axis=0)  # [H, W, anchors*dim]
        last_dim = out.shape[-1]
        per_anchor = last_dim // n_anchors
        reshaped = out.reshape(-1, per_anchor)
        if per_anchor == 4:
            loc_list.append(reshaped)
        elif per_anchor == 2:
            conf_list.append(reshaped)

    loc = np.concatenate(loc_list, axis=0)  # [N, 4]
    conf = np.concatenate(conf_list, axis=0)  # [N, 2]

    priors = generate_prior_boxes((h_model, w_model))

    boxes = decode_boxes(loc, priors)
    boxes *= np.array([w_model, h_model, w_model, h_model])

    scores = conf[:, 1]

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


class RetinaFace_MobileNetV1(ModelBase):
    """RetinaFace with MobileNetV1 backbone (folder: retinaface_mobilenet_v1_736x1280-1)."""

    info = ModelInfo(
        name="RetinaFace_MobileNetV1",
        dataset=DatasetType.widerface,
        evaluation=EvaluationType.widerface,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="default", size=(736, 1280)),
                Normalize(mean=[104.0, 117.0, 123.0], std=[1.0, 1.0, 1.0]),
                # Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="default", size=(736, 1280)),
                Normalize(mean=[104.0, 117.0, 123.0], std=[1.0, 1.0, 1.0]),
            ]
        )

    def postprocessing(self):
        return retinaface_v1_postprocessing
