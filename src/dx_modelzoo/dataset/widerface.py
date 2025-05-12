import os
from glob import glob
from typing import List

import cv2
from scipy.io import loadmat

from dx_modelzoo.dataset import DatasetBase


class WiderFaceDataset(DatasetBase):
    """WiderFace Dataset Class

    dataset_root's file tree should be like
        dataset_root/
            WIDER_val/
                images/
                    envent_a/
                        xxxxx.jpg
                        xxxxx.jpg
                    envent_b/
                        xxxxx.jpg
                        xxxxx.jpg
                    ...
            ground_truth/
                wider_easy_val.mat
                wider_face_val.mat
                wider_hard_val.mat
                wider_medium_val.mat

    Args:
        data_dir(str): dataset root dir.
    """

    def __init__(self, data_dir):
        super().__init__(data_dir)

        self.img_files: List[str] = sorted(glob(os.path.join(self.data_dir, "WIDER_val/images/**/*")))
        # self.gt_dir = os.path.join(self.data_dir, "ground_truth")
        # change to default folder name
        self.gt_dir = os.path.join(self.data_dir, "eval_tools/ground_truth")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        file_path = self.img_files[idx]
        img = cv2.imread(file_path)
        return self.preprocessing(img), img.shape, file_path

    def get_gt_boxes(self):
        """get gt boxes from mat files."""
        gt_mat = loadmat(os.path.join(self.gt_dir, "wider_face_val.mat"))
        hard_mat = loadmat(os.path.join(self.gt_dir, "wider_hard_val.mat"))
        medium_mat = loadmat(os.path.join(self.gt_dir, "wider_medium_val.mat"))
        easy_mat = loadmat(os.path.join(self.gt_dir, "wider_easy_val.mat"))

        facebox_list = gt_mat["face_bbx_list"]
        event_list = gt_mat["event_list"]
        file_list = gt_mat["file_list"]

        hard_gt_list = hard_mat["gt_list"]
        medium_gt_list = medium_mat["gt_list"]
        easy_gt_list = easy_mat["gt_list"]
        return (
            facebox_list,
            event_list,
            file_list,
            hard_gt_list,
            medium_gt_list,
            easy_gt_list,
        )
