import os

import h5py
import numpy as np

from dx_modelzoo.dataset import DatasetBase


class NYUDataset(DatasetBase):
    def __init__(self, data_dir: str) -> None:
        super().__init__(data_dir)

        classes, class_to_idx = self.find_classes()
        imgs = self.make_dataset(class_to_idx)

        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx

        self._depth_preprocessing = None

    @property
    def depth_preprocessing(self):
        if self._depth_preprocessing is None:
            raise ValueError("Depth Prerpocessing is not set.")
        return self._depth_preprocessing

    @depth_preprocessing.setter
    def depth_preprocessing(self, preprocessing) -> None:
        self._depth_preprocessing = preprocessing

    def h5_loader(self, path):
        h5f = h5py.File(path, "r")
        rgb = np.array(h5f["rgb"])
        rgb = np.transpose(rgb, (1, 2, 0))
        depth = np.array(h5f["depth"])
        return rgb, depth

    def find_classes(self):
        classes = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def is_image_file(self, filename):
        IMG_EXTENSIONS = [".h5"]
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

    def make_dataset(self, class_to_idx):
        images = []
        data_dir = os.path.expanduser(self.data_dir)
        for target in sorted(os.listdir(data_dir)):
            d = os.path.join(data_dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if self.is_image_file(fname):
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        images.append(item)
        return images

    def __getraw__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (rgb, depth) the raw data.
        """
        path, target = self.imgs[index]
        rgb, depth = self.h5_loader(path)
        return rgb, depth

    def __len__(self) -> None:
        return len(self.imgs)

    def __getitem__(self, index):
        rgb, depth = self.__getraw__(index)
        return self.preprocessing(rgb), self.depth_preprocessing(depth)
