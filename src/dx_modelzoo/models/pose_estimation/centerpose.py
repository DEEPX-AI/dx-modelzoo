"""CenterPose pose estimation models.

CenterNet-based multi-person pose estimation with heatmap outputs.
Outputs: person heatmap, center offset, keypoint offsets, bbox wh,
keypoint heatmaps, keypoint offset.
"""

from typing import List

import numpy as np
import torch
from torchvision.transforms import Compose

from dx_modelzoo.enums import DatasetType, EvaluationType
from dx_modelzoo.models import ModelBase, ModelInfo
from dx_modelzoo.preprocessing.convertcolor import ConvertColor
from dx_modelzoo.preprocessing.div import Div
from dx_modelzoo.preprocessing.normalize import Normalize
from dx_modelzoo.preprocessing.resize import Resize
from dx_modelzoo.preprocessing.transpose import Transpose


def _nms_heatmap(heatmap: np.ndarray, kernel: int = 3) -> np.ndarray:
    """Apply max-pooling based NMS on heatmap to find local peaks."""
    from scipy.ndimage import maximum_filter

    maxpool = maximum_filter(heatmap, size=kernel)
    keep = heatmap == maxpool
    return heatmap * keep


def centerpose_postprocessing(
    outputs: List[np.ndarray],
    input_size: int = 512,
    num_keypoints: int = 17,
    score_threshold: float = 0.001,
    top_k: int = 300,
):
    """CenterPose postprocessing.

    CenterPose outputs 6 heads (NCHW format):
        [0] hm:        [1, 1, H/4, W/4]   - person center heatmap
        [1] reg:       [1, 2, H/4, W/4]   - center offset (x, y)
        [2] hp:        [1, 34, H/4, W/4]  - keypoint offsets (17*2)
        [3] wh:        [1, 2, H/4, W/4]   - bbox width/height
        [4] hps:       [1, 17, H/4, W/4]  - keypoint heatmaps
        [5] hp_offset: [1, 2, H/4, W/4]   - keypoint center offset

    Args:
        outputs: list of numpy arrays from model inference.
        input_size: input image size.
        num_keypoints: number of keypoints (17 for COCO).
        score_threshold: minimum confidence threshold.
        top_k: maximum number of detections.

    Returns:
        tuple: (boxes_xyxy, scores, class_ids, keypoints) where
            boxes_xyxy: [N, 4] tensor
            scores: [N] tensor
            class_ids: [N] tensor (all zeros for single class)
            keypoints: [N, 17, 3] tensor (x, y, visibility)
    """
    empty = (
        torch.empty((0, 4)),
        torch.empty(0),
        torch.empty(0),
        torch.empty((0, num_keypoints, 3)),
    )

    if len(outputs) != 6:
        return empty

    # Model output order: hm[1,1,H,W], wh[1,2,H,W], hp[1,34,H,W],
    #                     reg[1,2,H,W], hps[1,17,H,W], hp_offset[1,2,H,W]
    hm = outputs[0][0, 0]  # [H, W]
    wh = outputs[1][0]  # [2, H, W]
    hp = outputs[2][0]  # [34, H, W]
    reg = outputs[3][0]  # [2, H, W]
    hps = outputs[4][0]  # [17, H, W]
    hp_offset = outputs[5][0]  # [2, H, W]

    H, W = hm.shape
    stride = input_size // H

    # Apply sigmoid to heatmap and NMS
    hm = 1.0 / (1.0 + np.exp(-hm))  # sigmoid
    hm = _nms_heatmap(hm)

    # Find top-k detections
    flat = hm.flatten()
    if len(flat) == 0:
        return empty

    k = min(top_k, len(flat))
    top_inds = np.argpartition(flat, -k)[-k:]
    top_scores = flat[top_inds]

    mask = top_scores > score_threshold
    top_inds = top_inds[mask]
    top_scores = top_scores[mask]

    if len(top_inds) == 0:
        return empty

    ys = top_inds // W
    xs = top_inds % W

    # Center offset
    cx = (xs.astype(np.float32) + reg[0, ys, xs]) * stride
    cy = (ys.astype(np.float32) + reg[1, ys, xs]) * stride

    # Bbox width/height
    bw = wh[0, ys, xs] * stride
    bh = wh[1, ys, xs] * stride

    x1 = np.clip(cx - bw / 2, 0, input_size)
    y1 = np.clip(cy - bh / 2, 0, input_size)
    x2 = np.clip(cx + bw / 2, 0, input_size)
    y2 = np.clip(cy + bh / 2, 0, input_size)

    # Keypoints: use center + offset approach
    kps = np.zeros((len(top_inds), num_keypoints, 3), dtype=np.float32)
    for j in range(num_keypoints):
        # Keypoint position = center + hp offset
        kp_x = cx + hp[j * 2, ys, xs] * stride
        kp_y = cy + hp[j * 2 + 1, ys, xs] * stride

        # Refine with keypoint heatmap
        # Find nearest peak in keypoint heatmap for visibility
        kp_hm = 1.0 / (1.0 + np.exp(-hps[j]))
        kp_grid_x = np.clip(np.round(kp_x / stride).astype(int), 0, W - 1)
        kp_grid_y = np.clip(np.round(kp_y / stride).astype(int), 0, H - 1)

        # Refine with hp_offset
        kp_x = kp_x + hp_offset[0, kp_grid_y, kp_grid_x] * stride
        kp_y = kp_y + hp_offset[1, kp_grid_y, kp_grid_x] * stride

        kps[:, j, 0] = np.clip(kp_x, 0, input_size)
        kps[:, j, 1] = np.clip(kp_y, 0, input_size)
        kps[:, j, 2] = kp_hm[kp_grid_y, kp_grid_x]

    boxes = torch.from_numpy(np.stack([x1, y1, x2, y2], axis=1)).float()
    scores_t = torch.from_numpy(top_scores).float()
    class_ids = torch.zeros(len(top_inds))
    keypoints_t = torch.from_numpy(kps).float()

    return boxes, scores_t, class_ids, keypoints_t


class CenterPose_RegNetX_1_6GF_FPN(ModelBase):
    """CenterPose with RegNetX-1.6GF-FPN backbone, 512x512 input."""

    info = ModelInfo(
        name="CenterPose_RegNetX_1_6GF_FPN",
        dataset=DatasetType.coco_pose,
        evaluation=EvaluationType.coco_pose,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=512, pad_location="edge", pad_value=[0, 0, 0]),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=512, pad_location="edge", pad_value=[0, 0, 0]),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return lambda outputs: centerpose_postprocessing(
            outputs,
            input_size=512,
            score_threshold=0.1,
        )


class CenterPose_RegNetX_800MF(ModelBase):
    """CenterPose with RegNetX-800MF backbone, 640x640 input."""

    info = ModelInfo(
        name="CenterPose_RegNetX_800MF",
        dataset=DatasetType.coco_pose,
        evaluation=EvaluationType.coco_pose,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=640, pad_location="edge", pad_value=[0, 0, 0]),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=640, pad_location="edge", pad_value=[0, 0, 0]),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return lambda outputs: centerpose_postprocessing(
            outputs,
            input_size=640,
            score_threshold=0.1,
        )


class CenterPose_RepVGG_A0(ModelBase):
    """CenterPose with RepVGG-A0 backbone, 416x416 input."""

    info = ModelInfo(
        name="CenterPose_RepVGG_A0",
        dataset=DatasetType.coco_pose,
        evaluation=EvaluationType.coco_pose,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=416, pad_location="edge", pad_value=[0, 0, 0]),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=416, pad_location="edge", pad_value=[0, 0, 0]),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return lambda outputs: centerpose_postprocessing(
            outputs,
            input_size=416,
            score_threshold=0.1,
        )
