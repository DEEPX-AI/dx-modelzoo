import os
from glob import glob
from typing import Dict, List, Tuple

import cv2
import numpy as np

from dx_modelzoo.dataset import DatasetBase


class ImageNetDataset(DatasetBase):
    """ImageNet Dataset Class.

    Read images from path and return as cv2.Image type.
    dataset_root's file tree should be like
        dataset_root/
            class_a/
                xxxxx.JPEG
                xxxxx.JPEG
                xxxxx.JPEG
                ...
            class_b/
                xxxxx.JPEG
                xxxxx.JPEG
                xxxxx.JPEG
                ...
            class_c/
                ...

    Args:
        data_dir (str): dataset root dir.
    """

    def __init__(self, data_dir: str):
        super().__init__(data_dir)

        self.image_files = sorted(glob(f"{self.data_dir}/**/*.JPEG", recursive=True))
        self._class_map = self._get_class_map()
        self.class_list = self._get_class_list()

    @property
    def class_map(self) -> Dict[str, int]:
        """Property of class map.

        Returns:
            Dict[str, int]: class map.
        """
        return self._class_map

    def _get_class_map(self) -> Dict[str, int]:
        """get class map from root dir.

        Returns:
            Dict[str, int]: _description_
        """
        dirs = [dir for dir in sorted(os.listdir(self.data_dir)) if os.path.isdir(os.path.join(self.data_dir, dir))]
        return {dir_name: idx for idx, dir_name in enumerate(dirs)}

    def _get_class_list(self) -> List[int]:
        """get class list.

        Returns:
            List[int]: class list.
        """
        class_list = []
        for image_path in self.image_files:
            class_name = image_path.split(os.path.sep)[-2]
            class_list.append(self.class_map[class_name])
        return class_list

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx) -> Tuple[np.ndarray, int]:
        img = cv2.imread(self.image_files[idx])
        img = self.preprocessing(img)
        label = self.class_list[idx]
        return img, label
