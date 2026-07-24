from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.dataset import DATASET_REGISTRY

_PANOPTIC_INSTALL_GUIDE = """\
  [BDD100K Panoptic Driving Perception] — Academic/Research use only

  Required structure under <DATA_ROOT>/bdd100k (or an absolute eval_path):
    bdd100k/
      images/100k/val/*.jpg                       (10k validation images)
      labels/det_20/det_val.json                  (object detection GT)
      labels/drivable/masks/val/*_drivable_id.png (drivable area masks {0,1,2})
      labels/lane/masks/val/*.png                 (lane line masks, binary)

  Sources (https://bdd-data.berkeley.edu/, registration required):
    - bdd100k_images_100k.zip
    - bdd100k_det_20_labels_trainval.zip
    - bdd100k_drivable_labels_trainval.zip
    - bdd100k_lane_labels_trainval.zip

  License: BSD 3-Clause. Requires registration.
"""

# YOLOPv2 "Traffic Object Detection" evaluates a single merged "vehicle" class.
_VEHICLE_CATEGORIES = ("car", "bus", "truck", "train")

_PANOPTIC_IMAGE_DIRS = [
    os.path.join("images", "100k", "val"),
    os.path.join("100k", "val"),
]
_PANOPTIC_DRIVABLE_DIRS = [
    os.path.join("labels", "drivable", "masks", "val"),
    os.path.join("labels", "val"),
]
_PANOPTIC_LANE_DIRS = [
    os.path.join("labels", "lane", "masks", "val"),
]
_PANOPTIC_DET_FILES = [
    os.path.join("labels", "det_20", "det_val.json"),
    os.path.join("labels", "det_val.json"),
]


def _first_existing_dir(data_dir: str, candidates: List[str]) -> str:
    for candidate in candidates:
        path = os.path.join(data_dir, candidate)
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(
        f"None of the expected directories found in {data_dir}: " f"{', '.join(candidates)}\n{_PANOPTIC_INSTALL_GUIDE}"
    )


@DATASET_REGISTRY.register
class BDD100K(DatasetBase):
    """BDD100K panoptic driving perception dataset.

    Provides ground truth for all three YOLOPv2 tasks simultaneously:
      * vehicle detection bounding boxes (single merged ``vehicle`` class),
      * drivable area segmentation (binary: non-drivable / drivable),
      * lane line segmentation (binary: background / lane).

    ``__getitem__`` returns ``(image, origin_shape, label)`` where ``label`` is
    a dict with keys ``boxes`` (``[N, 4]`` xyxy in original image coordinates),
    ``drivable`` (``[H, W]`` in ``{0, 1}``) and ``lane`` (``[H, W]`` in
    ``{0, 1}``).
    """

    num_class = 2

    def __init__(self, data_dir: str, vehicle_categories: Tuple[str, ...] = _VEHICLE_CATEGORIES) -> None:
        self.ensure_exists(data_dir, _PANOPTIC_INSTALL_GUIDE)
        super().__init__(data_dir)

        self.vehicle_categories = set(vehicle_categories)
        img_dir = _first_existing_dir(data_dir, _PANOPTIC_IMAGE_DIRS)
        drivable_dir = _first_existing_dir(data_dir, _PANOPTIC_DRIVABLE_DIRS)
        lane_dir = _first_existing_dir(data_dir, _PANOPTIC_LANE_DIRS)

        det_file = None
        for candidate in _PANOPTIC_DET_FILES:
            path = os.path.join(data_dir, candidate)
            if os.path.isfile(path):
                det_file = path
                break
        if det_file is None:
            raise FileNotFoundError(
                f"No detection label file found in {data_dir}. "
                f"Tried: {', '.join(_PANOPTIC_DET_FILES)}\n{_PANOPTIC_INSTALL_GUIDE}"
            )

        gt_boxes = self._load_detection_gt(det_file)

        # Build aligned sample list keyed by image stem; only keep samples that
        # have a matching drivable mask and lane mask on disk.
        self.img_paths: List[str] = []
        self.drivable_paths: List[str] = []
        self.lane_paths: List[str] = []
        self.gt_boxes: List[np.ndarray] = []

        for img_file in sorted(os.listdir(img_dir)):
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            stem = os.path.splitext(img_file)[0]
            drivable_path = os.path.join(drivable_dir, stem + "_drivable_id.png")
            lane_path = os.path.join(lane_dir, stem + ".png")
            if not (os.path.isfile(drivable_path) and os.path.isfile(lane_path)):
                continue
            self.img_paths.append(os.path.join(img_dir, img_file))
            self.drivable_paths.append(drivable_path)
            self.lane_paths.append(lane_path)
            self.gt_boxes.append(gt_boxes.get(img_file, np.zeros((0, 4), dtype=np.float32)))

        if not self.img_paths:
            raise FileNotFoundError(
                f"No aligned (image, drivable, lane) samples found in {data_dir}.\n{_PANOPTIC_INSTALL_GUIDE}"
            )

    def _load_detection_gt(self, det_file: str) -> Dict[str, np.ndarray]:
        with open(det_file, "r") as f:
            records = json.load(f)
        gt: Dict[str, np.ndarray] = {}
        for rec in records:
            boxes = []
            for lab in rec.get("labels") or []:
                if lab.get("category") not in self.vehicle_categories:
                    continue
                box = lab.get("box2d")
                if not box:
                    continue
                boxes.append([box["x1"], box["y1"], box["x2"], box["y2"]])
            gt[rec["name"]] = np.asarray(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
        return gt

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, index: int) -> Tuple:
        img = cv2.imread(self.img_paths[index])
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {self.img_paths[index]}")
        origin_shape = img.shape

        drivable = cv2.imread(self.drivable_paths[index], 0)
        if drivable is None:
            raise FileNotFoundError(f"Failed to load drivable mask: {self.drivable_paths[index]}")
        # Merge direct(1) + alternative(2) into drivable(1)
        drivable = np.where(drivable > 0, 1, 0).astype(np.int64)

        lane = cv2.imread(self.lane_paths[index], 0)
        if lane is None:
            raise FileNotFoundError(f"Failed to load lane mask: {self.lane_paths[index]}")
        lane = np.where(lane > 0, 1, 0).astype(np.int64)

        label = {
            "boxes": self.gt_boxes[index],
            "drivable": drivable,
            "lane": lane,
        }
        img = self.preprocessing(img)
        return img, origin_shape, label
