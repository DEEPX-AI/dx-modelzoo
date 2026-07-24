from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import Callable, List, Tuple

import cv2
import numpy as np
from PIL import Image

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.dataset import DATASET_REGISTRY

_INSTALL_GUIDE = """\
  [Pascal VOC 2007/2012] — "VOC2007" challenge data

  1. Download from http://host.robots.ox.ac.uk/pascal/VOC/voc2007/
     or Kaggle: https://www.kaggle.com/datasets/zaraks/pascal-voc-2007
  2. Extract under a ``PascalVOC/`` folder so the tree is
     ``<DATA_ROOT>/PascalVOC/VOCdevkit/VOC2007`` (and ``VOC2012``):
     Expected structure:
       PascalVOC/VOCdevkit/VOC2007/
         JPEGImages/
           000001.jpg ...
         SegmentationClass/
           000032.png ...
         Annotations/
           000001.xml ...
         ImageSets/
           Main/test.txt

       PascalVOC/VOCdevkit/VOC2012/
         JPEGImages/
           000001.jpg ...
         SegmentationClass/
           000032.png ...
         Annotations/
           000001.xml ...
         ImageSets/
           Main/test.txt
           Segmentation/val.txt
     NOTE: the dataset reads ``JPEGImages``/``SegmentationClass``/
     ``ImageSets`` directly under its ``eval_path``, so point ``eval_path``
     at a version dir (e.g. ``PascalVOC/VOCdevkit/VOC2012``).

  License: "VOC2007/VOC2012" challenge data — free for research use.
  See: http://host.robots.ox.ac.uk/pascal/VOC/voc2007/
"""


def get_image_label_path_list(data_dir: str) -> Tuple[List[str], List[str]]:
    image_dir = os.path.join(data_dir, "JPEGImages")
    labels_dir = os.path.join(data_dir, "SegmentationClass")
    file_name_txt = os.path.join(data_dir, "ImageSets", "Segmentation", "val.txt")
    with open(file_name_txt, "r") as f:
        file_names = f.read().splitlines()
    image_file_list = [os.path.join(image_dir, fn + ".jpg") for fn in file_names]
    label_file_list = [os.path.join(labels_dir, fn + ".png") for fn in file_names]
    return image_file_list, label_file_list


@DATASET_REGISTRY.register
class PascalVOC2012(DatasetBase):
    num_class = 21

    def __init__(self, data_dir: str) -> None:
        self.ensure_exists(data_dir, _INSTALL_GUIDE)
        super().__init__(data_dir)
        self.image_file_list, self.label_file_list = get_image_label_path_list(data_dir)
        self._label_preprocessing = None

    @property
    def label_preprocessing(self) -> Callable:
        if self._label_preprocessing is None:
            raise ValueError("Dataset's label preprocessing is not set.")
        return self._label_preprocessing

    @label_preprocessing.setter
    def label_preprocessing(self, value: Callable) -> None:
        self._label_preprocessing = value

    def __len__(self) -> int:
        return len(self.image_file_list)

    def __getitem__(self, idx: int) -> Tuple:
        img = cv2.imread(self.image_file_list[idx])
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {self.image_file_list[idx]}")
        img = self.preprocessing(img)
        label = np.array(Image.open(self.label_file_list[idx]))
        if self._label_preprocessing is not None:
            label = self.label_preprocessing(label)
        return img, label


@DATASET_REGISTRY.register
class PascalVOC2007(DatasetBase):
    def __init__(self, data_dir: str) -> None:
        self.ensure_exists(data_dir, _INSTALL_GUIDE)
        super().__init__(data_dir)
        self.data_ids = self._get_data_ids()
        self.class_names = (
            "BACKGROUND",
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        )
        self.class_dict = {name: i for i, name in enumerate(self.class_names)}
        self._group_annotation = self._get_group_annotation()

    @property
    def group_annotation(self) -> Tuple[dict, dict, dict]:
        return self._group_annotation

    @property
    def coco_annotation(self):
        """Build COCO-format annotation for unified evaluation."""
        if not hasattr(self, "_coco_annotation"):
            self._coco_annotation = self._build_coco_annotation()
        return self._coco_annotation

    def remap_class_id(self, model_class_id: int) -> int:
        """VOC uses direct class index — no remapping needed."""
        return int(model_class_id)

    def _build_coco_annotation(self):
        """Convert VOC XML annotations to COCO JSON format for pycocotools."""
        try:
            from faster_coco_eval import COCO as COCOApi
        except ImportError:
            return None

        images = []
        annotations = []
        categories = []
        ann_id = 1

        for class_index, class_name in enumerate(self.class_names):
            if class_index == 0:
                continue
            categories.append({"id": class_index, "name": class_name})

        for idx, data_id in enumerate(self.data_ids):
            img_path = os.path.join(self.data_dir, f"JPEGImages/{data_id}.jpg")
            img = cv2.imread(img_path)
            h, w = img.shape[:2] if img is not None else (0, 0)
            images.append({"id": idx, "file_name": f"{data_id}.jpg", "height": h, "width": w})

            gt_boxes, classes, is_difficult = self._parse_annotation(data_id)
            for i in range(len(gt_boxes)):
                x1, y1, x2, y2 = gt_boxes[i]
                ann = {
                    "id": ann_id,
                    "image_id": idx,
                    "category_id": int(classes[i]),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "area": float((x2 - x1) * (y2 - y1)),
                    "iscrowd": int(is_difficult[i]),
                }
                annotations.append(ann)
                ann_id += 1

        coco_ds = COCOApi()
        coco_ds.dataset = {"images": images, "annotations": annotations, "categories": categories}
        coco_ds.createIndex()
        return coco_ds

    def _get_data_ids(self) -> List[str]:
        ids_file = os.path.join(self.data_dir, "ImageSets/Main/test.txt")
        with open(ids_file) as f:
            return [line.rstrip() for line in f]

    def __getitem__(self, idx: int) -> Tuple:
        data_id = self.data_ids[idx]
        img_path = os.path.join(self.data_dir, f"JPEGImages/{data_id}.jpg")
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Failed to load image: {img_path}")
        return self.preprocessing(image), image.shape, idx

    def __len__(self) -> int:
        return len(self.data_ids)

    def _get_group_annotation(self) -> Tuple[dict, dict, dict]:
        true_case_stat = {}
        all_gt_boxes = {}
        all_difficult_cases = {}
        for image_id in self.data_ids:
            gt_boxes, classes, is_difficult = self._parse_annotation(image_id)
            for i, difficult in enumerate(is_difficult):
                class_index = int(classes[i])
                gt_box = gt_boxes[i]
                if not difficult:
                    true_case_stat[class_index] = true_case_stat.get(class_index, 0) + 1
                if class_index not in all_gt_boxes:
                    all_gt_boxes[class_index] = {}
                if image_id not in all_gt_boxes[class_index]:
                    all_gt_boxes[class_index][image_id] = []
                all_gt_boxes[class_index][image_id].append(gt_box)
                if class_index not in all_difficult_cases:
                    all_difficult_cases[class_index] = {}
                if image_id not in all_difficult_cases[class_index]:
                    all_difficult_cases[class_index][image_id] = []
                all_difficult_cases[class_index][image_id].append(difficult)
        for class_index in all_gt_boxes:
            for image_id in all_gt_boxes[class_index]:
                all_gt_boxes[class_index][image_id] = np.stack(all_gt_boxes[class_index][image_id])
        return true_case_stat, all_gt_boxes, all_difficult_cases

    def _parse_annotation(self, data_id: str) -> Tuple:
        annotation_file = os.path.join(self.data_dir, f"Annotations/{data_id}.xml")
        objects = ET.parse(annotation_file).findall("object")
        boxes, labels, is_difficult = [], [], []
        for obj in objects:
            class_name = obj.find("name").text.lower().strip()
            if class_name in self.class_dict:
                bbox = obj.find("bndbox")
                x1 = float(bbox.find("xmin").text) - 1
                y1 = float(bbox.find("ymin").text) - 1
                x2 = float(bbox.find("xmax").text) - 1
                y2 = float(bbox.find("ymax").text) - 1
                boxes.append([x1, y1, x2, y2])
                labels.append(self.class_dict[class_name])
                diff_str = obj.find("difficult").text
                is_difficult.append(int(diff_str) if diff_str else 0)
        return (
            np.array(boxes, dtype=np.float32),
            np.array(labels, dtype=np.int64),
            np.array(is_difficult, dtype=np.uint8),
        )
