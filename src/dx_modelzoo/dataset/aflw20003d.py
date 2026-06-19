from __future__ import annotations

import math
import os
from glob import glob
from typing import List, Tuple

import cv2
import numpy as np

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.dataset import DATASET_REGISTRY

_INSTALL_GUIDE = """\
  [AFLW2000-3D] — Research use only

  1. Download from http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm
     or: https://github.com/cleardusk/3DDFA
  2. Download AFLW2000-3D.zip
  3. Extract to: <DATA_ROOT>/AFLW20003D/
     Expected structure:
       AFLW20003D/
         image00002.jpg
         image00002.mat
         image00004.jpg
         image00004.mat
         ...

  Requires: scipy (for loading .mat files)
  License: Research use only.
  See: http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm
"""


@DATASET_REGISTRY.register
class AFLW20003D(DatasetBase):
    NUM_KEYPOINTS = 68

    def __init__(self, data_dir: str) -> None:
        self.ensure_exists(data_dir, _INSTALL_GUIDE)
        super().__init__(data_dir)
        self.samples: List[Tuple[str, str]] = []
        self._load_annotations()

    def _load_annotations(self) -> None:
        img_files = sorted(glob(os.path.join(self.data_dir, "*.jpg")))
        for img_path in img_files:
            mat_path = os.path.splitext(img_path)[0] + ".mat"
            if os.path.exists(mat_path):
                self.samples.append((img_path, mat_path))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple:
        try:
            import scipy.io as sio
        except ImportError:
            raise ImportError("scipy is required for AFLW2000-3D dataset")
        img_path, mat_path = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {img_path}")
        mat = sio.loadmat(mat_path)
        pt3d_68 = mat["pt3d_68"].astype(np.float32)
        pose_para = mat["Pose_Para"].flatten()
        yaw_deg = abs(math.degrees(pose_para[1]))
        roi = self._parse_roi_from_landmark(pt3d_68)
        face_crop = self._crop_img(img, roi)
        gt_img = pt3d_68[:2, :].T
        gt_landmarks = np.empty_like(gt_img)
        gt_landmarks[:, 0] = (gt_img[:, 0] - roi[0]) * 120.0 / (roi[2] - roi[0])
        gt_landmarks[:, 1] = (gt_img[:, 1] - roi[1]) * 120.0 / (roi[3] - roi[1])
        gt_w = gt_landmarks[:, 0].max() - gt_landmarks[:, 0].min()
        gt_h = gt_landmarks[:, 1].max() - gt_landmarks[:, 1].min()
        bbox_size = math.sqrt(max(gt_w, 1e-6) * max(gt_h, 1e-6))
        face_crop = self.preprocessing(face_crop)
        return face_crop, gt_landmarks, bbox_size, yaw_deg, idx

    @staticmethod
    def _parse_roi_from_landmark(pts: np.ndarray) -> list:
        bbox = [pts[0, :].min(), pts[1, :].min(), pts[0, :].max(), pts[1, :].max()]
        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        radius = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
        bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]
        llength = math.sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)
        cx = (bbox[2] + bbox[0]) / 2
        cy = (bbox[3] + bbox[1]) / 2
        return [cx - llength / 2, cy - llength / 2, cx + llength / 2, cy + llength / 2]

    @staticmethod
    def _crop_img(img: np.ndarray, roi: list) -> np.ndarray:
        h, w = img.shape[:2]
        sx, sy, ex, ey = [int(round(v)) for v in roi]
        dh, dw = ey - sy, ex - sx
        if dh <= 0 or dw <= 0:
            return img
        res = np.zeros((dh, dw, 3), dtype=np.uint8)
        dsx, dsy = max(0, -sx), max(0, -sy)
        dex, dey = dw - max(0, ex - w), dh - max(0, ey - h)
        src_sx, src_sy = max(0, sx), max(0, sy)
        src_ex, src_ey = min(w, ex), min(h, ey)
        if src_ex > src_sx and src_ey > src_sy:
            res[dsy:dey, dsx:dex] = img[src_sy:src_ey, src_sx:src_ex]
        return res
