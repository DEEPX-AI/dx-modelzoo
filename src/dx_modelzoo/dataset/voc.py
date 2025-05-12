import os
import xml.etree.ElementTree as ET
from typing import Callable, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from dx_modelzoo.dataset import DatasetBase


def get_image_label_path_list(data_dir: str) -> Tuple[List[str], List[str]]:
    """Get images and labels path list.
    Given a directory containing the image and label data, this function generates
    a list of file paths for the images and corresponding label files.

    Args:
        data_dir (str): The root directory where the dataset is stored. It should contain:
            - A "JPEGImages" directory with the images.
            - A "SegmentationClass" directory with the segmentation labels.
            - A "ImageSets/Segmentation/val.txt" file listing the image names (without extensions).

    Returns:
        Tuple[List[str], List[str]]:
            - A list of file paths for the images (".jpg" files).
            - A list of file paths for the corresponding labels (".png" files).

    Example:
        >>> image_paths, label_paths = get_image_label_path_list('/path/to/data')
        >>> print(image_paths[0])  # "/path/to/data/JPEGImages/image_1.jpg"
        >>> print(label_paths[0])  # "/path/to/data/SegmentationClass/image_1.png"
    """
    image_file_list = []
    label_file_list = []

    image_dir = os.path.join(data_dir, "JPEGImages")
    labels_dir = os.path.join(data_dir, "SegmentationClass")

    file_name_txt = os.path.join(data_dir, "ImageSets", "Segmentation", "val.txt")
    with open(file_name_txt, "r") as f:
        file_names = f.read().splitlines()
    image_file_list = [os.path.join(image_dir, file_name + ".jpg") for file_name in file_names]
    label_file_list = [os.path.join(labels_dir, file_name + ".png") for file_name in file_names]

    return image_file_list, label_file_list


class VOCSegmentation(DatasetBase):
    """VOC Segmentation Datasets.
    dataset_root's file tree should be like
        dataset_root/
            JPEGImages/
                ****.jpg
                ...
            SegmentationClass/
                ****.png
                ...
            ImageSets/
                Segmentation/
                    val.txt
    """

    num_class = 21

    def __init__(self, data_dir):
        super().__init__(data_dir)

        self.image_file_list, self.label_file_list = get_image_label_path_list(data_dir)
        self._label_preprocessing = None

    @property
    def label_preprocessing(self) -> Callable:
        if self._label_preprocessing is None:
            raise ValueError("Dataset's PreProcessing is not set.")
        return self._label_preprocessing

    @label_preprocessing.setter
    def label_preprocessing(self, lable_preprocessing: Callable) -> None:
        self._label_preprocessing = lable_preprocessing

    def __len__(self):
        return len(self.image_file_list)

    def __getitem__(self, idx):
        img = Image.open(self.image_file_list[idx])
        img = self.preprocessing(img)

        label = Image.open(self.label_file_list[idx])
        label = self.label_preprocessing(np.array(label))
        return img, label


class VOC2007Dection(DatasetBase):
    def __init__(self, data_dir):
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
        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}
        self._group_annotation = self._get_group_annotation()

    @property
    def group_annotation(self) -> Tuple[dict, dict, dict]:
        return self._group_annotation

    def _get_data_ids(self) -> List[str]:
        ids_file_path = os.path.join(self.data_dir, "ImageSets/Main/test.txt")
        with open(ids_file_path) as f:
            ids = [line.rstrip() for line in f]
        return ids

    def __getitem__(self, idx: int):
        data_id = self.data_ids[idx]
        image_file = os.path.join(self.data_dir, f"JPEGImages/{data_id}.jpg")
        image = cv2.imread(image_file)

        return self.preprocessing(image), image.shape

    def __len__(self):
        return len(self.data_ids)

    def _get_group_annotation(self) -> Tuple[dict, dict, dict]:
        true_case_stat = {}
        all_gt_boxes = {}
        all_difficult_cases = {}

        for image_id in self.data_ids:
            gt_boxes, classes, is_difficult = self._parse_annotation(image_id)

            gt_boxes = torch.from_numpy(gt_boxes)
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
                all_gt_boxes[class_index][image_id] = torch.stack(all_gt_boxes[class_index][image_id])
        for class_index in all_difficult_cases:
            for image_id in all_difficult_cases[class_index]:
                all_gt_boxes[class_index][image_id] = torch.tensor(all_gt_boxes[class_index][image_id])
        return true_case_stat, all_gt_boxes, all_difficult_cases

    def _parse_annotation(self, data_id: int) -> Tuple:
        annotation_file = os.path.join(self.data_dir, f"Annotations/{data_id}.xml")

        objects = ET.parse(annotation_file).findall("object")
        boxes = []
        labels = []
        is_difficult = []
        for object in objects:
            class_name = object.find("name").text.lower().strip()
            # we're only concerned with clases in our list
            if class_name in self.class_dict:
                bbox = object.find("bndbox")

                # VOC dataset format follows Matlab, in which indexes start from 0
                x1 = float(bbox.find("xmin").text) - 1
                y1 = float(bbox.find("ymin").text) - 1
                x2 = float(bbox.find("xmax").text) - 1
                y2 = float(bbox.find("ymax").text) - 1
                boxes.append([x1, y1, x2, y2])

                labels.append(self.class_dict[class_name])
                is_difficult_str = object.find("difficult").text
                is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)
        return (
            np.array(boxes, dtype=np.float32),
            np.array(labels, dtype=np.int64),
            np.array(is_difficult, dtype=np.uint8),
        )
