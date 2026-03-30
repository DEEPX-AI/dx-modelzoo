import math
import os
from glob import glob
from typing import List, Tuple

import cv2
import numpy as np
import scipy.io as sio

from dx_modelzoo.dataset import DatasetBase


class FaceLandmarkDataset(DatasetBase):
    """AFLW2000-3D dataset for face landmark evaluation.

    Reads image files paired with .mat annotation files containing
    68 3D face landmarks, bounding box, and pose parameters.

    Dataset structure:
        dataset_root/
            image00002.jpg
            image00002.mat
            ...

    Args:
        data_dir: Path to dataset root directory.
    """

    NUM_KEYPOINTS = 68

    def __init__(self, data_dir: str):
        super().__init__(data_dir)
        self.samples: List[Tuple[str, str]] = []
        self._load_annotations()

    def _load_annotations(self):
        """Find all image-mat pairs."""
        img_files = sorted(glob(os.path.join(self.data_dir, "*.jpg")))
        for img_path in img_files:
            basename = os.path.splitext(img_path)[0]
            mat_path = basename + ".mat"
            if os.path.exists(mat_path):
                self.samples.append((img_path, mat_path))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple:
        img_path, mat_path = self.samples[idx]

        img = cv2.imread(img_path)
        mat = sio.loadmat(mat_path)

        pt3d_68 = mat["pt3d_68"].astype(np.float32)  # [3, 68]

        # Yaw angle from Pose_Para (index 1 is yaw in radians)
        pose_para = mat["Pose_Para"].flatten()
        yaw_deg = abs(math.degrees(pose_para[1]))

        # Derive roi from GT landmarks (standard GT bbox init protocol)
        roi = self._parse_roi_from_landmark(pt3d_68)

        # Crop face using roi (zero-padded for out-of-bounds regions)
        face_crop = self._crop_img(img, roi)

        # GT landmarks in crop coordinate space (120x120)
        gt_img = pt3d_68[:2, :].T  # [68, 2] in image coords
        gt_landmarks = np.empty_like(gt_img)
        gt_landmarks[:, 0] = (gt_img[:, 0] - roi[0]) * 120.0 / (roi[2] - roi[0])
        gt_landmarks[:, 1] = (gt_img[:, 1] - roi[1]) * 120.0 / (roi[3] - roi[1])

        # Bbox size: sqrt(w * h) of GT landmark bbox in crop space
        gt_w = gt_landmarks[:, 0].max() - gt_landmarks[:, 0].min()
        gt_h = gt_landmarks[:, 1].max() - gt_landmarks[:, 1].min()
        bbox_size = math.sqrt(max(gt_w, 1e-6) * max(gt_h, 1e-6))

        face_crop = self.preprocessing(face_crop)

        return face_crop, gt_landmarks, bbox_size, yaw_deg, idx

    @staticmethod
    def _parse_roi_from_landmark(pts: np.ndarray) -> list:
        """Compute roi box from GT landmarks (3DDFA_V2 protocol)."""
        bbox = [pts[0, :].min(), pts[1, :].min(), pts[0, :].max(), pts[1, :].max()]
        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        radius = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
        bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]
        llength = math.sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)
        cx = (bbox[2] + bbox[0]) / 2
        cy = (bbox[3] + bbox[1]) / 2
        return [cx - llength / 2, cy - llength / 2, cx + llength / 2, cy + llength / 2]

    @staticmethod
    def _crop_img(img: np.ndarray, roi: np.ndarray) -> np.ndarray:
        """Crop image with zero-padding for out-of-bounds regions."""
        h, w = img.shape[:2]
        sx, sy, ex, ey = [int(round(v)) for v in roi]
        dh, dw = ey - sy, ex - sx
        if dh <= 0 or dw <= 0:
            return img

        res = np.zeros((dh, dw, 3), dtype=np.uint8)
        dsx = max(0, -sx)
        dsy = max(0, -sy)
        dex = dw - max(0, ex - w)
        dey = dh - max(0, ey - h)
        src_sx = max(0, sx)
        src_sy = max(0, sy)
        src_ex = min(w, ex)
        src_ey = min(h, ey)
        if src_ex > src_sx and src_ey > src_sy:
            res[dsy:dey, dsx:dex] = img[src_sy:src_ey, src_sx:src_ex]
        return res
