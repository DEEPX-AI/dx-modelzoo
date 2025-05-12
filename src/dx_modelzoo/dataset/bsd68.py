import os
from glob import glob

import cv2

from dx_modelzoo.dataset import DatasetBase


class BSD68Dataset(DatasetBase):
    """BSD68 Dataset Class.

    dataset_root's file tree should be like

        dataset_root/
            xxxxx.png
            xxxxx.png
            xxxxx.png
            ...
            xxxxx.png
    Args:
        data_dir (str): dataset_root.
    """

    def __init__(self, data_dir):
        super().__init__(data_dir)

        self.img_files = sorted(glob(os.path.join(self.data_dir, "*")))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        origin_img = cv2.imread(self.img_files[idx])
        return self.preprocessing(origin_img), origin_img
