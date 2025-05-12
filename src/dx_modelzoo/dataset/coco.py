import os
from copy import deepcopy
from glob import glob

import cv2
from pycocotools.coco import COCO

from dx_modelzoo.dataset import DatasetBase


class COCODataset(DatasetBase):
    """COCO datset.

    Read images from pat and return as cv2.Image type.
    dataset_root's file tree should be like
        dataset_root/
            val2017/
                coco_img_id.jpg
                ...
            anootations/
                instances_val2017.json
    for using pycocotools, need to instances_val2017.json

    Args:
        data_dir (str): dataset root dir.
    """

    def __init__(self, data_dir: str):
        super().__init__(data_dir)

        self.img_files = glob(os.path.join(data_dir, "images", "val2017", "*.jpg"))
        self.ids = list(map(lambda x: x.split(os.path.sep)[-1].split(".")[0], self.img_files))

        self.coco_annotation = COCO(os.path.join(data_dir, "annotations", "instances_val2017.json"))

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_files[idx])
        origin_img = deepcopy(img)
        img = self.preprocessing(img)
        return img, origin_img.shape, int(self.ids[idx])
