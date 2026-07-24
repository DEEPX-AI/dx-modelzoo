from __future__ import annotations

import math
import os
from glob import glob
from typing import List, Tuple

import cv2
import numpy as np

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.dataset import DATASET_REGISTRY

_INSTALL_GUIDE = """\
  [Hand Keypoints] — Custom dataset

  This dataset uses a custom hand keypoints format with YOLO-style labels.
  1. Prepare images and labels in the following structure:
  2. Extract to: <DATA_ROOT>/hand-keypoints/
     Expected structure:
       hand-keypoints/
         images/
           val/
             image_0001.jpg
             ...
         labels/
           val/
             image_0001.txt
             ...
     Each label file: <class> <cx> <cy> <w> <h> <kp1_x> <kp1_y> <kp1_v> ... (21 keypoints)

  Contact dataset maintainer for access.
"""

_WRIST = 0
_MIDDLE_FINGER_MCP = 9
_CROP_MARGIN = 1.1


@DATASET_REGISTRY.register
class HandKeypoints(DatasetBase):
    NUM_KEYPOINTS = 21

    def __init__(self, data_dir: str) -> None:
        self.ensure_exists(data_dir, _INSTALL_GUIDE)
        super().__init__(data_dir)
        self.samples: List[Tuple[str, np.ndarray, np.ndarray]] = []
        self._load_annotations()

    def _load_annotations(self) -> None:
        img_dir = os.path.join(self.data_dir, "images", "val")
        label_dir = os.path.join(self.data_dir, "labels", "val")
        img_files = sorted(glob(os.path.join(img_dir, "*.jpg")))
        if not img_files:
            img_files = sorted(glob(os.path.join(img_dir, "*.png")))
        for img_path in img_files:
            basename = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(label_dir, f"{basename}.txt")
            if not os.path.exists(label_path):
                continue
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5 + self.NUM_KEYPOINTS * 3:
                        continue
                    values = list(map(float, parts))
                    bbox = np.array(values[1:5], dtype=np.float32)
                    kpts = np.array(values[5 : 5 + self.NUM_KEYPOINTS * 3], dtype=np.float32).reshape(
                        self.NUM_KEYPOINTS, 3
                    )
                    self.samples.append((img_path, bbox, kpts))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple:
        img_path, bbox, kpts = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {img_path}")
        h, w = img.shape[:2]
        cx, cy, bw, bh = bbox
        center_px = np.array([cx * w, cy * h])
        gt_px = kpts[:, :2] * np.array([w, h])
        rotation_angle = self._compute_rotation(kpts, gt_px)
        diag = int(math.sqrt(w**2 + h**2))
        tx, ty = (diag - w) // 2, (diag - h) // 2
        M = cv2.getRotationMatrix2D(tuple(center_px), rotation_angle, 1.0)
        M[0, 2] += tx
        M[1, 2] += ty
        rotated = cv2.warpAffine(img, M, (diag, diag), borderValue=(0, 0, 0))
        new_center = center_px + np.array([tx, ty])
        side = max(bw * w, bh * h) * _CROP_MARGIN
        x1 = int(new_center[0] - side / 2)
        y1 = int(new_center[1] - side / 2)
        target = max(int(side), 1)
        crop = rotated[max(0, y1) : min(diag, y1 + target), max(0, x1) : min(diag, x1 + target)]
        pad_t, pad_b = max(0, -y1), max(0, (y1 + target) - diag)
        pad_l, pad_r = max(0, -x1), max(0, (x1 + target) - diag)
        if pad_t or pad_b or pad_l or pad_r:
            crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT)
        crop = crop[:target, :target]
        gt_h = np.hstack([gt_px, np.ones((self.NUM_KEYPOINTS, 1))])
        gt_rotated = (M @ gt_h.T).T[:, :2]
        gt_crop = (gt_rotated - np.array([x1, y1])) / side
        gt_kpts = np.zeros_like(kpts)
        gt_kpts[:, :2] = gt_crop
        gt_kpts[:, 2] = kpts[:, 2]
        crop = self.preprocessing(crop)
        return crop, gt_kpts.astype(np.float32), idx

    @staticmethod
    def _compute_rotation(kpts: np.ndarray, gt_px: np.ndarray) -> float:
        if kpts[_WRIST, 2] > 0 and kpts[_MIDDLE_FINGER_MCP, 2] > 0:
            d = gt_px[_MIDDLE_FINGER_MCP] - gt_px[_WRIST]
            angle = math.degrees(math.atan2(d[1], d[0]))
            return angle + 90.0
        return 0.0


@DATASET_REGISTRY.register
class HandKeypointsDetection(DatasetBase):
    """Whole-image hand *detection* view of the hand-keypoints dataset.

    Unlike :class:`HandKeypoints` (which crops a single rotated hand for
    landmark evaluation), this returns the full preprocessed image together
    with *all* hand bounding boxes present in the image, for detection (AP)
    evaluation.  Boxes are returned in normalized ``xyxy`` format.
    """

    def __init__(self, data_dir: str) -> None:
        self.ensure_exists(data_dir, _INSTALL_GUIDE)
        super().__init__(data_dir)
        self.samples: List[Tuple[str, np.ndarray]] = []
        self._load_detection_annotations()

    def _load_detection_annotations(self) -> None:
        img_dir = os.path.join(self.data_dir, "images", "val")
        label_dir = os.path.join(self.data_dir, "labels", "val")
        img_files = sorted(glob(os.path.join(img_dir, "*.jpg")))
        if not img_files:
            img_files = sorted(glob(os.path.join(img_dir, "*.png")))
        for img_path in img_files:
            basename = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(label_dir, f"{basename}.txt")
            boxes: List[List[float]] = []
            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        _cls, cx, cy, bw, bh = map(float, parts[:5])
                        boxes.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
            self.samples.append((img_path, np.asarray(boxes, dtype=np.float32).reshape(-1, 4)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple:
        img_path, boxes = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {img_path}")
        h, w = img.shape[:2]
        image = self.preprocessing(img)
        return image, boxes, (h, w), img_path
