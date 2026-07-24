from __future__ import annotations

import os
from glob import glob
from typing import List, Tuple

import cv2

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.dataset import DATASET_REGISTRY

_INSTALL_GUIDE = """\
  [WiderFace] — Non-commercial research only

  1. Download from http://shuoyang1213.me/WIDERFACE/
     - WIDER_val.zip (validation images)
     - Face Annotations (.mat format)
     - Evaluation tools
  2. Extract to: <DATA_ROOT>/widerface/
     Expected structure:
       widerface/
         WIDER_val/
           images/
             0--Parade/
               0_Parade_marchingband_1_5.jpg
               ...
         eval_tools/
           ground_truth/
             wider_face_val.mat
             wider_easy_val.mat
             wider_medium_val.mat
             wider_hard_val.mat

  License: Non-commercial research only.
  See: http://shuoyang1213.me/WIDERFACE/
"""


@DATASET_REGISTRY.register
class WiderFace(DatasetBase):
    def __init__(self, data_dir: str) -> None:
        self.ensure_exists(data_dir, _INSTALL_GUIDE)
        super().__init__(data_dir)
        self.img_files: List[str] = sorted(glob(os.path.join(self.data_dir, "WIDER_val/images/**/*")))
        self.gt_dir = os.path.join(self.data_dir, "eval_tools/ground_truth")

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, idx: int) -> Tuple:
        file_path = self.img_files[idx]
        img = cv2.imread(file_path)
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {file_path}")
        return self.preprocessing(img), img.shape, file_path

    def get_gt_boxes(self):
        try:
            from scipy.io import loadmat
        except ImportError:
            raise ImportError("scipy is required for WiderFace evaluation")
        gt_mat = loadmat(os.path.join(self.gt_dir, "wider_face_val.mat"))
        hard_mat = loadmat(os.path.join(self.gt_dir, "wider_hard_val.mat"))
        medium_mat = loadmat(os.path.join(self.gt_dir, "wider_medium_val.mat"))
        easy_mat = loadmat(os.path.join(self.gt_dir, "wider_easy_val.mat"))
        return (
            gt_mat["face_bbx_list"],
            gt_mat["event_list"],
            gt_mat["file_list"],
            hard_mat["gt_list"],
            medium_mat["gt_list"],
            easy_mat["gt_list"],
        )
