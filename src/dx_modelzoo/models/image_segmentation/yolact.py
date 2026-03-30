import numpy as np
import torch
import torchvision
from torchvision.transforms import Compose

from dx_modelzoo.enums import DatasetType, EvaluationType
from dx_modelzoo.evaluator import EvaluatorBase
from dx_modelzoo.models import ModelBase, ModelInfo
from dx_modelzoo.preprocessing.convertcolor import ConvertColor
from dx_modelzoo.preprocessing.div import Div
from dx_modelzoo.preprocessing.resize import Resize
from dx_modelzoo.preprocessing.transpose import Transpose

# Cache for prior boxes (avoid regenerating every frame)
_prior_cache = {}


def yolact_postprocessing(predictions, conf_thresh=0.001, nms_thresh=0.7, top_k=300, max_num_detections=100):
    """
    YOLACT postprocessing for NPU outputs.

    Args:
        predictions: List of 4 NPU outputs [loc, conf, mask_coeffs, proto]
            - loc: (1, N, 4) box offsets
            - conf: (1, N, 81) class scores (including background)
            - mask_coeffs: (1, N, 32) mask coefficients
            - proto: (1, H, W, 32) prototype masks in NHWC format

    Returns:
        [outputs, proto] where:
            - outputs: torch.Tensor (M, 38) [x1,y1,x2,y2, score, class_id, mask_coeffs(32)]
            - proto: torch.Tensor (32, H, W) prototype masks
    """
    if not isinstance(predictions, list) or len(predictions) not in [4, 5]:
        raise ValueError(
            f"Expected list of 4 outputs, got {type(predictions)} with "
            f"len {len(predictions) if isinstance(predictions, list) else 'N/A'}"
        )

    if len(predictions) == 5:
        loc, conf, mask_coeffs, _, proto = predictions
    else:
        loc, conf, mask_coeffs, proto = predictions

    # Get input size from proto shape (proto is NHWC: 1, H, W, 32)
    input_size = proto.shape[1] * 4  # proto is 1/4 of input size

    # Convert proto from NHWC to CHW (remove batch dim) - use contiguous for efficiency
    proto_out = torch.from_numpy(np.ascontiguousarray(proto[0].transpose(2, 0, 1)))

    # Pre-filter by confidence before decoding - stay in numpy for speed
    conf_0 = conf[0, :, 1:]  # (N, 80) exclude background - single indexing
    max_scores = conf_0.max(axis=1)  # (N,)
    class_ids = conf_0.argmax(axis=1)  # (N,)

    # Confidence threshold filter
    keep_mask = max_scores > conf_thresh
    if not keep_mask.any():
        return [torch.zeros((0, 38), dtype=torch.float32), proto_out]

    keep_indices = np.nonzero(keep_mask)[0]

    # Top-k before decoding
    if len(keep_indices) > top_k:
        topk_idx = np.argpartition(max_scores[keep_indices], -top_k)[-top_k:]
        keep_indices = keep_indices[topk_idx]

    # Generate priors and decode only selected boxes
    priors = _generate_yolact_priors(input_size)
    boxes = _decode_boxes_fast(loc[0, keep_indices], priors[keep_indices])  # (K, 4) normalized xyxy

    # Run NMS on filtered results
    result = _yolact_nms(
        boxes,  # (K, 4) normalized xyxy
        max_scores[keep_indices],  # (K,)
        class_ids[keep_indices],  # (K,)
        mask_coeffs[0, keep_indices],  # (K, 32)
        input_size=input_size,
        nms_thresh=nms_thresh,
        max_num_detections=max_num_detections,
    )

    return [torch.from_numpy(result), proto_out]


def _decode_boxes_fast(loc, priors):
    """
    Decode YOLACT predicted bbox coordinates (optimized version).

    Args:
        loc: (K, 4) predicted offsets [dx, dy, dw, dh]
        priors: (K, 4) prior boxes in center-size format [cx, cy, w, h] (normalized 0-1)

    Returns:
        (K, 4) decoded boxes in xyxy format (normalized 0-1)
    """
    variances = (0.1, 0.2)

    # Decode center and size
    cx = priors[:, 0] + loc[:, 0] * variances[0] * priors[:, 2]
    cy = priors[:, 1] + loc[:, 1] * variances[0] * priors[:, 3]
    w = priors[:, 2] * np.exp(np.clip(loc[:, 2] * variances[1], -10, 10))
    h = priors[:, 3] * np.exp(np.clip(loc[:, 3] * variances[1], -10, 10))

    # Convert from center-size to xyxy format
    half_w = w * 0.5
    half_h = h * 0.5
    boxes = np.column_stack(
        [
            np.clip(cx - half_w, 0, 1),
            np.clip(cy - half_h, 0, 1),
            np.clip(cx + half_w, 0, 1),
            np.clip(cy + half_h, 0, 1),
        ]
    )
    return boxes.astype(np.float32)


def _yolact_nms(
    boxes,
    scores,
    class_ids,
    mask_coeffs,
    input_size,
    nms_thresh=0.7,
    max_num_detections=300,
):
    """Run NMS on pre-filtered YOLACT outputs.

    Args:
        boxes: (K, 4) decoded xyxy coordinates (normalized 0-1)
        scores: (K,) max class scores (already filtered by conf_thresh and top_k)
        class_ids: (K,) class ids
        mask_coeffs: (K, 32) mask coefficients
        input_size: input image size for scaling boxes

    Returns:
        np.ndarray (M, 38): [x1,y1,x2,y2, score, class_id, mask_coeffs(32)]
    """
    n = len(boxes)
    if n == 0:
        return np.zeros((0, 38), dtype=np.float32)

    # Scale boxes to input_size for NMS - avoid copy if possible
    boxes_scaled = boxes * input_size

    # Batched NMS (per-class NMS)
    nms_idx = torchvision.ops.batched_nms(
        torch.from_numpy(boxes_scaled),
        torch.from_numpy(scores.astype(np.float32)),
        torch.from_numpy(class_ids.astype(np.int64)),
        nms_thresh,
    )

    if nms_idx.size(0) == 0:
        return np.zeros((0, 38), dtype=np.float32)

    idx = nms_idx[:max_num_detections].numpy()

    # Pre-allocate result array
    m = len(idx)
    result = np.empty((m, 38), dtype=np.float32)
    result[:, :4] = boxes_scaled[idx]
    result[:, 4] = scores[idx]
    result[:, 5] = class_ids[idx]
    result[:, 6:] = mask_coeffs[idx]

    return result


def _generate_yolact_priors(input_size=512):
    """Generate YOLACT prior boxes with caching.

    Original YOLACT make_priors iteration order:
        for j, i in product(range(conv_h), range(conv_w)):  # row-major (y outer, x inner)
            for ars in aspect_ratios:
                for scale in scales:
                    for ar in ars:
                        ...

    For YOLACT with 5 FPN levels, each level has:
    - scales: [scale * 2^(0/3), scale * 2^(1/3), scale * 2^(2/3)] (3 octave scales)
    - aspect_ratios: [[1, 0.5, 2]] (3 ARs)

    Total per level: 3 scales * 3 ARs = 9 anchors per position
    Total: 64^2*9 + 32^2*9 + 16^2*9 + 8^2*9 + 4^2*9 = 49104 priors
    """
    global _prior_cache

    if input_size in _prior_cache:
        return _prior_cache[input_size]

    from math import sqrt

    base_scales = [24, 48, 96, 192, 384]
    aspect_ratios = [1, 0.5, 2]
    strides = [8, 16, 32, 64, 128]

    priors = []
    for stride, base_scale in zip(strides, base_scales):
        conv_size = input_size // stride
        scales = [base_scale * 2 ** (j / 3.0) for j in range(3)]

        # Original YOLACT order: spatial (row-major), then scale, then AR
        for j in range(conv_size):  # row (y) - OUTER
            for i in range(conv_size):  # col (x) - INNER
                cx = (i + 0.5) / conv_size
                cy = (j + 0.5) / conv_size

                for scale in scales:
                    for ar in aspect_ratios:
                        ar_sqrt = sqrt(ar)
                        w = scale / input_size * ar_sqrt
                        h = scale / input_size / ar_sqrt
                        priors.append([cx, cy, w, h])

    result = np.array(priors, dtype=np.float32)
    _prior_cache[input_size] = result
    return result


class YOLACT_RegNetX_1_6gf(ModelBase):
    info = ModelInfo("YOLACT_RegNetX_1_6gf", dataset=DatasetType.coco, evaluation=EvaluationType.instance_segmentation)

    def __init__(self, evaluator: EvaluatorBase) -> None:
        super().__init__(evaluator)
        self.evaluator.lazy_postprocessing = False

    def preprocessing(self):
        return Compose(
            [
                ConvertColor("BGR2RGB"),
                Resize(mode="pad", size=512, pad_location="edge", pad_value=[114, 114, 114]),
                Div(255),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                ConvertColor("BGR2RGB"),
                Resize(mode="pad", size=512, pad_location="edge", pad_value=[114, 114, 114]),
            ]
        )

    def postprocessing(self):
        return yolact_postprocessing


class YOLACT_RegNetX_800mf(ModelBase):
    info = ModelInfo("YOLACT_RegNetX_800mf", dataset=DatasetType.coco, evaluation=EvaluationType.instance_segmentation)

    def __init__(self, evaluator: EvaluatorBase) -> None:
        super().__init__(evaluator)
        self.evaluator.lazy_postprocessing = False

    def preprocessing(self):
        return Compose(
            [
                ConvertColor("BGR2RGB"),
                Resize(mode="pad", size=512, pad_location="edge", pad_value=[114, 114, 114]),
                Div(255),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                ConvertColor("BGR2RGB"),
                Resize(mode="pad", size=512, pad_location="edge", pad_value=[114, 114, 114]),
            ]
        )

    def postprocessing(self):
        return yolact_postprocessing
