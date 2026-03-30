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

try:
    from torchvision.ops import nms as tv_nms
except Exception:
    tv_nms = None


def nanodet_postprocessing(
    outputs: List[np.ndarray],
    input_size: int = 416,
    num_class: int = 80,
    reg_max: int = 10,
    strides: List[int] = [8, 16, 32],
    score_threshold: float = 0.001,
    nms_threshold: float = 0.7,
) -> torch.Tensor:
    if len(outputs) == 0:
        return torch.empty((0, 6), dtype=torch.float32)

    feats = outputs[0]

    # accept torch.Tensor or numpy array
    if isinstance(feats, torch.Tensor):
        feats = feats.detach().cpu().numpy()

    # squeeze/reshape like before
    if feats.ndim == 3:
        if feats.shape[0] == 1:
            feats = feats.squeeze(0)
        elif feats.shape[1] == 1:
            feats = feats.squeeze(1)

    if feats.ndim > 2:
        feats = feats.reshape(-1, feats.shape[-1])

    num_points = feats.shape[0]
    expected_channels = num_class + 4 * (reg_max + 1)
    if feats.shape[1] != expected_channels:
        return torch.empty((0, 6), dtype=torch.float32)

    center_priors = generate_grid_center_priors(input_size, input_size, strides)
    if len(center_priors) != num_points:
        return torch.empty((0, 6), dtype=torch.float32)

    # scores and label selection (vectorized)
    scores_all = feats[:, :num_class].astype(np.float32)
    max_scores = np.max(scores_all, axis=1)
    labels = np.argmax(scores_all, axis=1)

    keep_mask = max_scores > score_threshold
    if not np.any(keep_mask):
        return torch.empty((0, 6), dtype=torch.float32)

    sel_idxs = np.nonzero(keep_mask)[0]
    sel_scores = max_scores[sel_idxs].astype(np.float32)
    sel_labels = labels[sel_idxs].astype(np.int64)
    sel_centers = center_priors[sel_idxs].astype(np.float32)  # (n,3)

    # bbox distribution preds reshape -> (n, 4, reg_max+1)
    bbox_preds = feats[sel_idxs, num_class:].astype(np.float32)
    n = bbox_preds.shape[0]
    bbox_preds = bbox_preds.reshape(n, 4, reg_max + 1)

    # vector softmax along last dim
    x = bbox_preds
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    prob = exp_x / (np.sum(exp_x, axis=-1, keepdims=True) + 1e-12)  # (n,4,reg_max+1)

    coef = np.arange(reg_max + 1, dtype=np.float32)
    # expectation sum(j * p_j)
    dis = np.sum(prob * coef.reshape(1, 1, -1), axis=-1)  # (n,4)

    strides_sel = sel_centers[:, 2].reshape(-1, 1)  # (n,1)
    dis = dis * strides_sel  # scale by stride

    ct_x = (sel_centers[:, 0] + 0.5) * strides_sel[:, 0]
    ct_y = (sel_centers[:, 1] + 0.5) * strides_sel[:, 0]

    x1 = np.clip(ct_x - dis[:, 0], 0.0, float(input_size))
    y1 = np.clip(ct_y - dis[:, 1], 0.0, float(input_size))
    x2 = np.clip(ct_x + dis[:, 2], 0.0, float(input_size))
    y2 = np.clip(ct_y + dis[:, 3], 0.0, float(input_size))

    boxes_np = np.stack([x1, y1, x2, y2], axis=1)  # (n,4)

    # Class-wise NMS (use torchvision.ops.nms if available)
    keep_indices = []

    if tv_nms is not None:
        boxes_t = torch.from_numpy(boxes_np).float()
        scores_t = torch.from_numpy(sel_scores).float()
        labels_np = sel_labels
        # iterate unique classes (only on selected preds)
        unique_labels = np.unique(labels_np)
        for cls in unique_labels:
            cls_mask = np.nonzero(labels_np == cls)[0]
            cls_boxes = boxes_t[cls_mask]
            cls_scores = scores_t[cls_mask]
            if cls_boxes.numel() == 0:
                continue
            keep = tv_nms(cls_boxes, cls_scores, nms_threshold)
            keep_indices.extend(sel_idxs[cls_mask[keep.numpy()]].tolist())
    else:
        # fallback to Python NMS for each class (uses nms_boxes defined above)
        from collections import defaultdict

        cmap = defaultdict(list)
        for i_local, i_global in enumerate(sel_idxs):
            b = {
                "x1": float(boxes_np[i_local, 0]),
                "y1": float(boxes_np[i_local, 1]),
                "x2": float(boxes_np[i_local, 2]),
                "y2": float(boxes_np[i_local, 3]),
                "score": float(sel_scores[i_local]),
                "label": int(sel_labels[i_local]),
            }
            cmap[int(sel_labels[i_local])].append((i_global, b))
        for cls, items in cmap.items():
            boxes_list = [b for (_, b) in items]
            kept = nms_boxes(boxes_list, nms_threshold)
            kept_global_idxs = []
            # map back to sel_idxs indices
            # score_to_globals = {
            #     (b["x1"], b["y1"], b["x2"], b["y2"], b["score"]): idx for idx, b in [(it[0], it[1]) for it in items]
            # }
            for kb in kept:
                # key = (kb["x1"], kb["y1"], kb["x2"], kb["y2"], kb["score"])
                # find matching global index (best-effort)
                for i_local, (i_global, orig_b) in enumerate(items):
                    if (
                        abs(orig_b["x1"] - kb["x1"]) < 1e-6
                        and abs(orig_b["y1"] - kb["y1"]) < 1e-6
                        and abs(orig_b["x2"] - kb["x2"]) < 1e-6
                        and abs(orig_b["y2"] - kb["y2"]) < 1e-6
                        and abs(orig_b["score"] - kb["score"]) < 1e-6
                    ):
                        kept_global_idxs.append(i_global)
                        break
            keep_indices.extend(kept_global_idxs)

    if len(keep_indices) == 0:
        return torch.empty((0, 6), dtype=torch.float32)

    # keep_indices are global indices relative to original num_points
    # we need to build final arrays for those indices
    final_boxes = []
    for gi in keep_indices:
        # gi is an index into the original feats points; map to sel_idxs position
        pos = np.where(sel_idxs == gi)[0]
        if pos.size == 0:
            continue
        i_local = int(pos[0])
        final_boxes.append(
            [
                float(boxes_np[i_local, 0]),
                float(boxes_np[i_local, 1]),
                float(boxes_np[i_local, 2]),
                float(boxes_np[i_local, 3]),
                float(sel_scores[i_local]),
                float(sel_labels[i_local]),
            ]
        )

    if len(final_boxes) == 0:
        return torch.empty((0, 6), dtype=torch.float32)

    result_tensor = torch.tensor(final_boxes, dtype=torch.float32)
    return result_tensor


def generate_grid_center_priors(input_height: int, input_width: int, strides: List[int]) -> np.ndarray:
    """Generate center priors in format of (x, y, stride).

    Official NanoDet implementation from NCNN demo.
    """
    center_priors = []
    for stride in strides:
        feat_w = int(np.ceil(input_width / stride))
        feat_h = int(np.ceil(input_height / stride))
        for y in range(feat_h):
            for x in range(feat_w):
                center_priors.append([x, y, stride])
    return np.array(center_priors, dtype=np.float32)


def nanodet_repvgg_postprocessing(
    outputs: List[np.ndarray],
    input_size: int = 640,
    num_class: int = 80,
    score_threshold: float = 0.001,
    nms_threshold: float = 0.7,
) -> torch.Tensor:
    """Postprocessing for NanoDet_RepVGGA models with FCOS-style output format.

    Output format: [1, 8400, 85] where 85 = 4 (bbox) + 1 (obj_conf) + 80 (class_scores)
    Box decoding (YOLOX-style):
        cx = (raw_cx + grid_x) * stride
        cy = (raw_cy + grid_y) * stride
        w  = exp(raw_w) * stride
        h  = exp(raw_h) * stride

    Grid: 8400 = 80*80 (stride 8) + 40*40 (stride 16) + 20*20 (stride 32)
    """
    if len(outputs) == 0:
        return torch.empty((0, 6), dtype=torch.float32)

    feats = outputs[0]

    # accept torch.Tensor or numpy array
    if isinstance(feats, torch.Tensor):
        feats = feats.detach().cpu().numpy()

    # squeeze batch dimension if present
    if feats.ndim == 3 and feats.shape[0] == 1:
        feats = feats.squeeze(0)  # [8400, 85]

    if feats.ndim != 2:
        return torch.empty((0, 6), dtype=torch.float32)

    # num_boxes = feats.shape[0]
    num_channels = feats.shape[1]

    if num_channels != 85:
        return torch.empty((0, 6), dtype=torch.float32)

    # Generate grid indices for each stride (YOLOX uses integer grid indices)
    strides = [8, 16, 32]
    grid_indices = []
    grid_strides = []
    for stride in strides:
        gs = input_size // stride
        for gy in range(gs):
            for gx in range(gs):
                grid_indices.append((gx, gy))
                grid_strides.append(stride)

    grid_indices = np.array(grid_indices, dtype=np.float32)  # [8400, 2]
    grid_strides = np.array(grid_strides, dtype=np.float32)  # [8400]

    # Parse output: [cx_raw, cy_raw, w_log, h_log] (4) + obj_conf (1) + class_scores (80)
    # YOLOX decode: center = (raw + grid_idx) * stride, size = exp(log_wh) * stride
    box_raw = feats[:, :4]  # [n, 4]
    obj_conf = feats[:, 4:5]  # [n, 1] - already sigmoid applied in ONNX
    class_scores = feats[:, 5:]  # [n, 80] - already sigmoid applied in ONNX

    # Compute final scores: obj_conf * class_score
    scores = obj_conf * class_scores  # [n, 80]

    # Get max class scores and labels
    max_scores = np.max(scores, axis=1)  # [n]
    labels = np.argmax(scores, axis=1)  # [n]

    # Filter by score threshold
    keep_mask = max_scores > score_threshold
    if not np.any(keep_mask):
        return torch.empty((0, 6), dtype=torch.float32)

    sel_idxs = np.nonzero(keep_mask)[0]
    sel_scores = max_scores[sel_idxs].astype(np.float32)
    sel_labels = labels[sel_idxs].astype(np.int64)
    sel_box = box_raw[sel_idxs]
    sel_grids = grid_indices[sel_idxs]
    sel_strides = grid_strides[sel_idxs]

    # Decode boxes: YOLOX-style — cx = (raw + grid_x) * stride
    cx = (sel_box[:, 0] + sel_grids[:, 0]) * sel_strides
    cy = (sel_box[:, 1] + sel_grids[:, 1]) * sel_strides
    bw = np.exp(np.clip(sel_box[:, 2], -50.0, 50.0)) * sel_strides
    bh = np.exp(np.clip(sel_box[:, 3], -50.0, 50.0)) * sel_strides

    # Convert cxcywh to xyxy
    x1 = np.clip(cx - bw / 2, 0.0, float(input_size))
    y1 = np.clip(cy - bh / 2, 0.0, float(input_size))
    x2 = np.clip(cx + bw / 2, 0.0, float(input_size))
    y2 = np.clip(cy + bh / 2, 0.0, float(input_size))

    boxes_np = np.stack([x1, y1, x2, y2], axis=1)  # (n, 4)

    # Class-wise NMS
    keep_indices = []
    if tv_nms is not None:
        boxes_t = torch.from_numpy(boxes_np).float()
        scores_t = torch.from_numpy(sel_scores).float()
        unique_labels = np.unique(sel_labels)
        for cls in unique_labels:
            cls_mask = np.nonzero(sel_labels == cls)[0]
            cls_boxes = boxes_t[cls_mask]
            cls_scores = scores_t[cls_mask]
            if cls_boxes.numel() == 0:
                continue
            keep = tv_nms(cls_boxes, cls_scores, nms_threshold)
            keep_indices.extend(cls_mask[keep.numpy()].tolist())
    else:
        # Fallback NMS
        for cls in np.unique(sel_labels):
            cls_mask = np.nonzero(sel_labels == cls)[0]
            cls_boxes = [
                {
                    "x1": boxes_np[i, 0],
                    "y1": boxes_np[i, 1],
                    "x2": boxes_np[i, 2],
                    "y2": boxes_np[i, 3],
                    "score": sel_scores[i],
                    "idx": i,
                }
                for i in cls_mask
            ]
            cls_boxes = sorted(cls_boxes, key=lambda x: x["score"], reverse=True)
            kept = []
            while cls_boxes:
                kept.append(cls_boxes[0]["idx"])
                if len(cls_boxes) == 1:
                    break
                base = cls_boxes[0]
                remaining = []
                for b in cls_boxes[1:]:
                    xx1 = max(base["x1"], b["x1"])
                    yy1 = max(base["y1"], b["y1"])
                    xx2 = min(base["x2"], b["x2"])
                    yy2 = min(base["y2"], b["y2"])
                    inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
                    area1 = (base["x2"] - base["x1"]) * (base["y2"] - base["y1"])
                    area2 = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
                    iou = inter / (area1 + area2 - inter + 1e-10)
                    if iou < nms_threshold:
                        remaining.append(b)
                cls_boxes = remaining
            keep_indices.extend(kept)

    if len(keep_indices) == 0:
        return torch.empty((0, 6), dtype=torch.float32)

    # Build final result
    final_boxes = []
    for i in keep_indices:
        final_boxes.append(
            [
                float(boxes_np[i, 0]),
                float(boxes_np[i, 1]),
                float(boxes_np[i, 2]),
                float(boxes_np[i, 3]),
                float(sel_scores[i]),
                float(sel_labels[i]),
            ]
        )

    return torch.tensor(final_boxes, dtype=torch.float32)


def nms_boxes(boxes: List[dict], nms_threshold: float = 0.7) -> List[dict]:
    """Apply NMS to boxes."""
    if len(boxes) == 0:
        return []

    boxes = sorted(boxes, key=lambda x: x["score"], reverse=True)
    areas = [(b["x2"] - b["x1"]) * (b["y2"] - b["y1"]) for b in boxes]

    keep = []
    while len(boxes) > 0:
        keep.append(boxes[0])
        if len(boxes) == 1:
            break

        remaining = []
        remaining_areas = []
        for i in range(1, len(boxes)):
            xx1 = max(boxes[0]["x1"], boxes[i]["x1"])
            yy1 = max(boxes[0]["y1"], boxes[i]["y1"])
            xx2 = min(boxes[0]["x2"], boxes[i]["x2"])
            yy2 = min(boxes[0]["y2"], boxes[i]["y2"])

            w = max(0.0, xx2 - xx1)
            h = max(0.0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[0] + areas[i] - inter + 1e-10)

            if iou < nms_threshold:
                remaining.append(boxes[i])
                remaining_areas.append(areas[i])

        boxes = remaining
        areas = remaining_areas

    return keep


class NanoDet(ModelBase):
    info = ModelInfo(
        name="NanoDet",
        dataset=DatasetType.coco,
        evaluation=EvaluationType.coco,
        raw_performance="27.8",  # NanoDet-RepVGG-A0 416x416 mAP
        q_lite_performance="",
    )
    input_size = 416
    num_class = 80
    reg_max = 10
    strides = [8, 16, 32]

    def __init__(self, evaluator):
        self.input_size = 416
        self.num_class = 80
        self.reg_max = 10
        self.strides = [8, 16, 32]
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=self.input_size, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor(form="BGR2RGB"),  # RGB → BGR conversion (BGR2RGB is bidirectional)
                Normalize(mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=self.input_size, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor(form="BGR2RGB"),  # RGB → BGR conversion for NPU as well
            ]
        )

    def postprocessing(self):
        input_size = self.input_size
        num_class = self.num_class
        reg_max = self.reg_max
        strides = self.strides

        def _post(outputs: List[np.ndarray]):

            result = nanodet_postprocessing(
                outputs,
                input_size=input_size,
                num_class=num_class,
                reg_max=reg_max,
                strides=strides,
                score_threshold=0.001,  # Lowered: sigmoid already applied in ONNX
                nms_threshold=0.7,
            )

            return result

        return _post


class NanoDet_RepVGGA(ModelBase):
    info = ModelInfo(
        name="NanoDet_RepVGGA",
        dataset=DatasetType.coco,
        evaluation=EvaluationType.coco,
        raw_performance="27.8",  # NanoDet-RepVGG-A0 416x416 mAP
        q_lite_performance="",
    )
    input_size = 640
    num_class = 80
    reg_max = 10
    strides = [8, 16, 32]

    def __init__(self, evaluator):
        self.input_size = 640
        self.num_class = 80
        self.reg_max = 10
        self.strides = [8, 16, 32]
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=self.input_size, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor(form="BGR2RGB"),  # RGB → BGR conversion (BGR2RGB is bidirectional)
                Normalize(mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=self.input_size, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor(form="BGR2RGB"),  # RGB → BGR conversion for NPU as well
            ]
        )

    def postprocessing(self):
        input_size = self.input_size
        num_class = self.num_class
        reg_max = self.reg_max
        strides = self.strides

        def _post(outputs: List[np.ndarray]):

            result = nanodet_postprocessing(
                outputs,
                input_size=input_size,
                num_class=num_class,
                reg_max=reg_max,
                strides=strides,
                score_threshold=0.001,  # Lowered: sigmoid already applied in ONNX
                nms_threshold=0.7,
            )

            return result

        return _post


class NanoDet_Plus(ModelBase):
    """NanoDet-Plus-M model with GFL head."""

    info = ModelInfo(
        name="NanoDet_Plus",
        dataset=DatasetType.coco,
        evaluation=EvaluationType.coco,
    )
    input_size = 416
    num_class = 80
    reg_max = 7  # NanoDet-Plus uses reg_max=7
    strides = [8, 16, 32, 64]  # 4 scales

    def __init__(self, evaluator):
        self.input_size = 416
        self.num_class = 80
        self.reg_max = 7
        self.strides = [8, 16, 32, 64]
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=self.input_size, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor(form="BGR2RGB"),
                Normalize(mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=self.input_size, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor(form="BGR2RGB"),
            ]
        )

    def postprocessing(self):
        input_size = self.input_size
        num_class = self.num_class
        reg_max = self.reg_max
        strides = self.strides

        def _post(outputs: List[np.ndarray]):
            result = nanodet_postprocessing(
                outputs,
                input_size=input_size,
                num_class=num_class,
                reg_max=reg_max,
                strides=strides,
                score_threshold=0.001,
                nms_threshold=0.7,
            )
            return result

        return _post


class NanoDet_Plus_15(ModelBase):
    """NanoDet-Plus-M 1.5x model with GFL head."""

    info = ModelInfo(
        name="NanoDet_Plus_15",
        dataset=DatasetType.coco,
        evaluation=EvaluationType.coco,
    )
    input_size = 416
    num_class = 80
    reg_max = 7
    strides = [8, 16, 32, 64]

    def __init__(self, evaluator):
        self.input_size = 416
        self.num_class = 80
        self.reg_max = 7
        self.strides = [8, 16, 32, 64]
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=self.input_size, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor(form="BGR2RGB"),
                Normalize(mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=self.input_size, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor(form="BGR2RGB"),
            ]
        )

    def postprocessing(self):
        input_size = self.input_size
        num_class = self.num_class
        reg_max = self.reg_max
        strides = self.strides

        def _post(outputs: List[np.ndarray]):
            result = nanodet_postprocessing(
                outputs,
                input_size=input_size,
                num_class=num_class,
                reg_max=reg_max,
                strides=strides,
                score_threshold=0.001,
                nms_threshold=0.7,
            )
            return result

        return _post


class NanoDet_RepVGGA12(ModelBase):
    """NanoDet-RepVGGA12 model with YOLOX-style output format."""

    info = ModelInfo(
        name="NanoDet_RepVGGA12",
        dataset=DatasetType.coco,
        evaluation=EvaluationType.coco,
    )
    input_size = 640
    num_class = 80
    reg_max = 10
    strides = [8, 16, 32]

    def __init__(self, evaluator):
        self.input_size = 640
        self.num_class = 80
        self.reg_max = 10
        self.strides = [8, 16, 32]
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=self.input_size, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor(form="BGR2RGB"),
                Div(1.0),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=self.input_size, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor(form="BGR2RGB"),
            ]
        )

    def postprocessing(self):
        input_size = self.input_size
        num_class = self.num_class

        def _post(outputs: List[np.ndarray]):
            return nanodet_repvgg_postprocessing(
                outputs,
                input_size=input_size,
                num_class=num_class,
                score_threshold=0.001,
                nms_threshold=0.7,
            )

        return _post
