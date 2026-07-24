from __future__ import annotations

import os
from glob import glob
from typing import Tuple

import cv2
import numpy as np

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.dataset import DATASET_REGISTRY

_INSTALL_GUIDE = """\
  [COCO 2017] — CC BY 4.0 (free for research and commercial use)

  1. Download from https://cocodataset.org/#download
     - val2017.zip (validation images, ~1GB)
     - annotations_trainval2017.zip (annotations)
  2. Extract to: <DATA_ROOT>/COCO/
     Expected structure:
       COCO/
         val2017/
             000000000139.jpg
             ...
         annotations/
             instances_val2017.json
             person_keypoints_val2017.json
     NOTE: All COCO-based datasets (COCO, COCOPose, COCOPoseTopDown,
     COCOPersonSeg) share this layout.

  License: CC BY 4.0 — free for research and commercial use.
  See: https://cocodataset.org/#termsofuse
"""


@DATASET_REGISTRY.register
class COCO(DatasetBase):
    """COCO dataset.

    Expected structure:
        data_dir/
            val2017/*.jpg
            annotations/instances_val2017.json
    """

    def __init__(self, data_dir: str) -> None:
        self.ensure_exists(data_dir, _INSTALL_GUIDE)
        super().__init__(data_dir)
        self.img_files = sorted(glob(os.path.join(data_dir, "val2017", "*.jpg")))
        self.ids = [os.path.splitext(os.path.basename(f))[0] for f in self.img_files]

        # Optional: load COCO annotations if faster_coco_eval available
        self._coco = None
        ann_file = os.path.join(data_dir, "annotations", "instances_val2017.json")
        if os.path.exists(ann_file):
            try:
                from faster_coco_eval import COCO

                self._coco = COCO(ann_file)
            except ImportError:
                pass

    @property
    def coco_annotation(self):
        return self._coco

    def remap_class_id(self, model_class_id: int) -> int:
        """Map 80-class model output to 91-class COCO category ID."""
        from dx_modelzoo.evaluator.constant import COCO80TO91MAPPER

        return COCO80TO91MAPPER.get(model_class_id, model_class_id)

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Tuple[int, ...], int]:
        img = cv2.imread(self.img_files[idx])
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {self.img_files[idx]}")
        origin_shape = img.shape
        img = self.preprocessing(img)
        return img, origin_shape, int(self.ids[idx])


@DATASET_REGISTRY.register
class COCOPose(DatasetBase):
    def __init__(self, data_dir: str) -> None:
        self.ensure_exists(data_dir, _INSTALL_GUIDE)
        super().__init__(data_dir)
        self.img_files = sorted(glob(os.path.join(data_dir, "val2017", "*.jpg")))
        self.ids = [os.path.splitext(os.path.basename(f))[0] for f in self.img_files]
        self._coco = None
        ann_file = os.path.join(data_dir, "annotations", "person_keypoints_val2017.json")
        if os.path.exists(ann_file):
            try:
                from faster_coco_eval import COCO

                self._coco = COCO(ann_file)
            except ImportError:
                pass

    @property
    def coco_annotation(self):
        return self._coco

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, idx: int) -> Tuple:
        img = cv2.imread(self.img_files[idx])
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {self.img_files[idx]}")
        origin_shape = img.shape
        img = self.preprocessing(img)
        return img, origin_shape, int(self.ids[idx])


@DATASET_REGISTRY.register
class COCOPoseTopDown(DatasetBase):
    """COCO top-down pose dataset.

    Each sample is a single person crop from a COCO validation image,
    using GT bounding boxes with 25% padding. Returns preprocessed crops
    along with keypoints transformed into crop coordinates.

    Expected structure:
        data_dir/
            val2017/*.jpg
            annotations/person_keypoints_val2017.json
    """

    _BBOX_PAD_RATIO = 0.25

    def __init__(self, data_dir: str, bbox_pad_ratio: float = 0.25, square_crop: bool = False) -> None:
        self.ensure_exists(data_dir, _INSTALL_GUIDE)
        super().__init__(data_dir)
        self.data_dir = data_dir
        self._bbox_pad_ratio = bbox_pad_ratio
        self._square_crop = square_crop

        ann_file = os.path.join(data_dir, "annotations", "person_keypoints_val2017.json")
        from faster_coco_eval import COCO

        self._coco = COCO(ann_file)

        # Build per-person sample list: (img_id, ann) for annotations with >=1 visible keypoint
        self.samples: list[Tuple[int, dict]] = []
        for ann in self._coco.dataset["annotations"]:
            if ann.get("iscrowd", 0):
                continue
            kpts = ann.get("keypoints", [])
            if len(kpts) < 3:
                continue
            # visibility flags are at indices 2, 5, 8, ... (every 3rd starting from index 2)
            vis = kpts[2::3]
            if any(v > 0 for v in vis):
                self.samples.append((ann["image_id"], ann))

    @property
    def coco_annotation(self):
        return self._coco

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple:
        img_id, ann = self.samples[idx]
        ann_id = ann["id"]

        # Load full image
        img_info = self._coco.imgs[img_id]
        img_path = os.path.join(self.data_dir, "val2017", img_info["file_name"])
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {img_path}")
        img_h, img_w = img.shape[:2]

        # Extract and pad bbox (COCO format: [x, y, w, h])
        bx, by, bw, bh = ann["bbox"]
        pad_w = bw * self._bbox_pad_ratio
        pad_h = bh * self._bbox_pad_ratio

        if self._square_crop:
            # Make crop square centered on bbox center
            cx, cy = bx + bw / 2, by + bh / 2
            side = max(bw + 2 * pad_w, bh + 2 * pad_h)
            x1 = max(0, int(cx - side / 2))
            y1 = max(0, int(cy - side / 2))
            x2 = min(img_w, int(cx + side / 2))
            y2 = min(img_h, int(cy + side / 2))
        else:
            x1 = max(0, int(bx - pad_w))
            y1 = max(0, int(by - pad_h))
            x2 = min(img_w, int(bx + bw + pad_w))
            y2 = min(img_h, int(by + bh + pad_h))

        crop = img[y1:y2, x1:x2]
        crop_h, crop_w = crop.shape[:2]

        # Parse keypoints [x1, y1, v1, x2, y2, v2, ...] -> coordinates and visibility
        raw_kpts = ann["keypoints"]
        kpts_full = np.array(raw_kpts, dtype=np.float32).reshape(17, 3)
        visibility = kpts_full[:, 2].astype(np.int32)  # [17]

        # Transform keypoints from full-image coords to crop coords,
        # then scale to model input size (192 x 256)
        model_w, model_h = 192, 256
        kpts_crop = np.zeros((17, 2), dtype=np.float32)
        if crop_w > 0 and crop_h > 0:
            kpts_crop[:, 0] = (kpts_full[:, 0] - x1) * (model_w / crop_w)
            kpts_crop[:, 1] = (kpts_full[:, 1] - y1) * (model_h / crop_h)

        # Original COCO bbox for evaluation output
        gt_bbox = (bx, by, bw, bh)
        # Crop origin and size for back-mapping predicted keypoints to image coords
        crop_params = (x1, y1, crop_w, crop_h)

        # Preprocess the crop (resize, normalize, etc.)
        crop = self.preprocessing(crop)

        return crop, gt_bbox, crop_params, kpts_crop, visibility, img_id, ann_id


@DATASET_REGISTRY.register
class COCOPersonSeg(DatasetBase):
    """COCO person-vs-background semantic segmentation dataset.

    Builds GT by rasterizing the union of all ``person`` instance
    segmentations (polygons + crowd RLE) from the COCO annotations into a
    binary semantic mask (0=bg, 1=person) resized to the model input size.

    Expected structure:
        data_dir/
            val2017/*.jpg
            annotations/person_keypoints_val2017.json  (or instances_val2017.json)
    """

    num_class = 2

    def __init__(
        self, data_dir: str, inputs=None, ann_file: str | None = None, person_only: bool = True, **kwargs
    ) -> None:
        self.ensure_exists(data_dir, _INSTALL_GUIDE)
        super().__init__(data_dir)
        from faster_coco_eval import COCO as PyCOCO

        if ann_file is None:
            cand = [
                os.path.join(data_dir, "annotations", "person_keypoints_val2017.json"),
                os.path.join(data_dir, "annotations", "instances_val2017.json"),
            ]
            ann_file = next((c for c in cand if os.path.exists(c)), cand[0])
        self.coco = PyCOCO(ann_file)
        self.person_cat = 1  # COCO 'person' category id
        self.img_dir = os.path.join(data_dir, "val2017")
        if person_only:
            ids = self.coco.getImgIds(catIds=[self.person_cat])
        else:
            ids = self.coco.getImgIds()
        self.img_ids = sorted(set(ids))
        self.in_h, self.in_w = self._infer_input_hw(inputs)

    @staticmethod
    def _infer_input_hw(inputs) -> Tuple[int, int]:
        """Return model input (H, W) from the YAML inputs spec (NCHW assumed)."""
        if inputs:
            shape = inputs[0].get("shape") if isinstance(inputs[0], dict) else None
            if shape and len(shape) == 4:
                return int(shape[2]), int(shape[3])
        return (512, 512)

    def __len__(self) -> int:
        return len(self.img_ids)

    def _person_mask(self, img_id: int, h0: int, w0: int) -> np.ndarray:
        mask = np.zeros((h0, w0), dtype=np.uint8)
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=[self.person_cat], iscrowd=None)
        for ann in self.coco.loadAnns(ann_ids):
            m = self.coco.annToMask(ann)
            if m.shape != (h0, w0):
                m = cv2.resize(m, (w0, h0), interpolation=cv2.INTER_NEAREST)
            mask |= m.astype(np.uint8)
        return mask

    def __getitem__(self, idx: int):
        info = self.coco.loadImgs(self.img_ids[idx])[0]
        img_path = os.path.join(self.img_dir, info["file_name"])
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {img_path}")
        h0, w0 = img.shape[:2]
        image = self.preprocessing(img)
        mask = self._person_mask(int(info["id"]), h0, w0)
        label = cv2.resize(mask, (self.in_w, self.in_h), interpolation=cv2.INTER_NEAREST)
        return image, label.astype(np.int64)
