import os

import cv2
import numpy as np

from dx_modelzoo.dataset import DatasetBase

labels_info = [
    {
        "hasInstances": False,
        "category": "void",
        "catid": 0,
        "name": "unlabeled",
        "ignoreInEval": True,
        "id": 0,
        "color": [0, 0, 0],
        "trainId": 255,
    },
    {
        "hasInstances": False,
        "category": "void",
        "catid": 0,
        "name": "ego vehicle",
        "ignoreInEval": True,
        "id": 1,
        "color": [0, 0, 0],
        "trainId": 255,
    },
    {
        "hasInstances": False,
        "category": "void",
        "catid": 0,
        "name": "rectification border",
        "ignoreInEval": True,
        "id": 2,
        "color": [0, 0, 0],
        "trainId": 255,
    },
    {
        "hasInstances": False,
        "category": "void",
        "catid": 0,
        "name": "out of roi",
        "ignoreInEval": True,
        "id": 3,
        "color": [0, 0, 0],
        "trainId": 255,
    },
    {
        "hasInstances": False,
        "category": "void",
        "catid": 0,
        "name": "static",
        "ignoreInEval": True,
        "id": 4,
        "color": [0, 0, 0],
        "trainId": 255,
    },
    {
        "hasInstances": False,
        "category": "void",
        "catid": 0,
        "name": "dynamic",
        "ignoreInEval": True,
        "id": 5,
        "color": [111, 74, 0],
        "trainId": 255,
    },
    {
        "hasInstances": False,
        "category": "void",
        "catid": 0,
        "name": "ground",
        "ignoreInEval": True,
        "id": 6,
        "color": [81, 0, 81],
        "trainId": 255,
    },
    {
        "hasInstances": False,
        "category": "flat",
        "catid": 1,
        "name": "road",
        "ignoreInEval": False,
        "id": 7,
        "color": [128, 64, 128],
        "trainId": 0,
    },
    {
        "hasInstances": False,
        "category": "flat",
        "catid": 1,
        "name": "sidewalk",
        "ignoreInEval": False,
        "id": 8,
        "color": [244, 35, 232],
        "trainId": 1,
    },
    {
        "hasInstances": False,
        "category": "flat",
        "catid": 1,
        "name": "parking",
        "ignoreInEval": True,
        "id": 9,
        "color": [250, 170, 160],
        "trainId": 255,
    },
    {
        "hasInstances": False,
        "category": "flat",
        "catid": 1,
        "name": "rail track",
        "ignoreInEval": True,
        "id": 10,
        "color": [230, 150, 140],
        "trainId": 255,
    },
    {
        "hasInstances": False,
        "category": "construction",
        "catid": 2,
        "name": "building",
        "ignoreInEval": False,
        "id": 11,
        "color": [70, 70, 70],
        "trainId": 2,
    },
    {
        "hasInstances": False,
        "category": "construction",
        "catid": 2,
        "name": "wall",
        "ignoreInEval": False,
        "id": 12,
        "color": [102, 102, 156],
        "trainId": 3,
    },
    {
        "hasInstances": False,
        "category": "construction",
        "catid": 2,
        "name": "fence",
        "ignoreInEval": False,
        "id": 13,
        "color": [190, 153, 153],
        "trainId": 4,
    },
    {
        "hasInstances": False,
        "category": "construction",
        "catid": 2,
        "name": "guard rail",
        "ignoreInEval": True,
        "id": 14,
        "color": [180, 165, 180],
        "trainId": 255,
    },
    {
        "hasInstances": False,
        "category": "construction",
        "catid": 2,
        "name": "bridge",
        "ignoreInEval": True,
        "id": 15,
        "color": [150, 100, 100],
        "trainId": 255,
    },
    {
        "hasInstances": False,
        "category": "construction",
        "catid": 2,
        "name": "tunnel",
        "ignoreInEval": True,
        "id": 16,
        "color": [150, 120, 90],
        "trainId": 255,
    },
    {
        "hasInstances": False,
        "category": "object",
        "catid": 3,
        "name": "pole",
        "ignoreInEval": False,
        "id": 17,
        "color": [153, 153, 153],
        "trainId": 5,
    },
    {
        "hasInstances": False,
        "category": "object",
        "catid": 3,
        "name": "polegroup",
        "ignoreInEval": True,
        "id": 18,
        "color": [153, 153, 153],
        "trainId": 255,
    },
    {
        "hasInstances": False,
        "category": "object",
        "catid": 3,
        "name": "traffic light",
        "ignoreInEval": False,
        "id": 19,
        "color": [250, 170, 30],
        "trainId": 6,
    },
    {
        "hasInstances": False,
        "category": "object",
        "catid": 3,
        "name": "traffic sign",
        "ignoreInEval": False,
        "id": 20,
        "color": [220, 220, 0],
        "trainId": 7,
    },
    {
        "hasInstances": False,
        "category": "nature",
        "catid": 4,
        "name": "vegetation",
        "ignoreInEval": False,
        "id": 21,
        "color": [107, 142, 35],
        "trainId": 8,
    },
    {
        "hasInstances": False,
        "category": "nature",
        "catid": 4,
        "name": "terrain",
        "ignoreInEval": False,
        "id": 22,
        "color": [152, 251, 152],
        "trainId": 9,
    },
    {
        "hasInstances": False,
        "category": "sky",
        "catid": 5,
        "name": "sky",
        "ignoreInEval": False,
        "id": 23,
        "color": [70, 130, 180],
        "trainId": 10,
    },
    {
        "hasInstances": True,
        "category": "human",
        "catid": 6,
        "name": "person",
        "ignoreInEval": False,
        "id": 24,
        "color": [220, 20, 60],
        "trainId": 11,
    },
    {
        "hasInstances": True,
        "category": "human",
        "catid": 6,
        "name": "rider",
        "ignoreInEval": False,
        "id": 25,
        "color": [255, 0, 0],
        "trainId": 12,
    },
    {
        "hasInstances": True,
        "category": "vehicle",
        "catid": 7,
        "name": "car",
        "ignoreInEval": False,
        "id": 26,
        "color": [0, 0, 142],
        "trainId": 13,
    },
    {
        "hasInstances": True,
        "category": "vehicle",
        "catid": 7,
        "name": "truck",
        "ignoreInEval": False,
        "id": 27,
        "color": [0, 0, 70],
        "trainId": 14,
    },
    {
        "hasInstances": True,
        "category": "vehicle",
        "catid": 7,
        "name": "bus",
        "ignoreInEval": False,
        "id": 28,
        "color": [0, 60, 100],
        "trainId": 15,
    },
    {
        "hasInstances": True,
        "category": "vehicle",
        "catid": 7,
        "name": "caravan",
        "ignoreInEval": True,
        "id": 29,
        "color": [0, 0, 90],
        "trainId": 255,
    },
    {
        "hasInstances": True,
        "category": "vehicle",
        "catid": 7,
        "name": "trailer",
        "ignoreInEval": True,
        "id": 30,
        "color": [0, 0, 110],
        "trainId": 255,
    },
    {
        "hasInstances": True,
        "category": "vehicle",
        "catid": 7,
        "name": "train",
        "ignoreInEval": False,
        "id": 31,
        "color": [0, 80, 100],
        "trainId": 16,
    },
    {
        "hasInstances": True,
        "category": "vehicle",
        "catid": 7,
        "name": "motorcycle",
        "ignoreInEval": False,
        "id": 32,
        "color": [0, 0, 230],
        "trainId": 17,
    },
    {
        "hasInstances": True,
        "category": "vehicle",
        "catid": 7,
        "name": "bicycle",
        "ignoreInEval": False,
        "id": 33,
        "color": [119, 11, 32],
        "trainId": 18,
    },
    {
        "hasInstances": False,
        "category": "vehicle",
        "catid": 7,
        "name": "license plate",
        "ignoreInEval": True,
        "id": -1,
        "color": [0, 0, 142],
        "trainId": -1,
    },
]


class CitySpaceDataset(DatasetBase):
    """CityScpace Dataset Class.

    dataset_root's file tree should be like
        dataset_root/
            val.txt
            leftImg8bit/
                val/
                    category_a/
                        xxxx.png
                        xxxx.png
                        ...
                    category_b/
                        xxxx.png
            gtFine/
                val/
                    category_a/
                        xxxx.png
                        xxxx.png
                    category_b/
                        xxxx.png

    Args:
        data_dir(str): dataset root dir.
    """

    def __init__(self, data_dir):
        super().__init__(data_dir)

        self.annpath = f"{self.data_dir}/val.txt"

        with open(self.annpath, "r") as fr:
            pairs = fr.read().splitlines()
        self.img_paths, self.lb_paths = [], []
        for pair in pairs:
            imgpth, lbpth = pair.split(",")
            self.img_paths.append(os.path.join(self.data_dir, imgpth))
            self.lb_paths.append(os.path.join(self.data_dir, lbpth))

        assert len(self.img_paths) == len(self.lb_paths)
        self.len = len(self.img_paths)

        self.num_class = 19
        self.lb_ignore = 255
        self.lb_map = np.arange(256).astype(np.uint8)
        for el in labels_info:
            self.lb_map[el["id"]] = el["trainId"]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        impth, lbpth = self.img_paths[index], self.lb_paths[index]
        img, label = self.get_image(impth, lbpth)
        if self.lb_map is not None:
            label = self.lb_map[label]

        img = self.preprocessing(img)
        return img, label

    def get_image(self, impth, lbpth):
        img = cv2.imread(impth).copy()
        label = cv2.imread(lbpth, 0)
        return img, label
