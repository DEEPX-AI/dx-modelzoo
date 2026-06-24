"""KITTI 3D object detection dataset (BEV input for SFA3D).

The model consumes a Bird's-Eye-View (BEV) map generated from the LiDAR point
cloud, so this dataset turns each ``velodyne/*.bin`` frame into a 3-channel
``[intensity, height, density]`` BEV image. Ground-truth 3D boxes are read from
``label_2`` and projected into LiDAR coordinates with the per-frame ``calib``.

Expected layout (standard KITTI 3D object detection)::

    KITTI/
      training/
        velodyne/000000.bin
        label_2/000000.txt
        calib/000000.txt
      ImageSets/
        val.txt          # one frame id per line (e.g. 000123)

The BEV map is returned as an HWC uint8 image in [0, 255] so the generic YAML
preprocessing (resize/div 255/transpose/expanddim) reproduces the model input.

BEV config and the camera->LiDAR math mirror the SFA3D reference
(https://github.com/maudzung/SFA3D).
"""
from __future__ import annotations

import os
from typing import Tuple

import numpy as np

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.dataset import DATASET_REGISTRY

_INSTALL_GUIDE = """\
  [KITTI 3D Object Detection]

  1. Download from https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d
       - Velodyne point clouds (29 GB)
       - training labels
       - camera calibration matrices
  2. Extract to: <DATA_ROOT>/KITTI/
     Expected structure:
       KITTI/
         training/
           velodyne/000000.bin ...
           label_2/000000.txt ...
           calib/000000.txt ...
         ImageSets/
           val.txt
  3. Provide an eval split list at ImageSets/val.txt (one 6-digit id per line).
     The common val split (3769 frames) is widely available, e.g.
     https://raw.githubusercontent.com/maudzung/SFA3D/master/dataset/kitti/ImageSets/val.txt

  License: KITTI is for non-commercial research use.
"""

# 0=Pedestrian, 1=Car, 2=Cyclist  (Van->Car, Person_sitting->Pedestrian)
CLASS_NAME_TO_ID = {
    "Pedestrian": 0,
    "Car": 1,
    "Cyclist": 2,
    "Van": 1,
    "Person_sitting": 0,
}
CLASS_NAMES = ["Pedestrian", "Car", "Cyclist"]

# BEV / boundary config (SFA3D kitti_config)
BOUNDARY = {"minX": 0.0, "maxX": 50.0, "minY": -25.0, "maxY": 25.0, "minZ": -2.73, "maxZ": 1.27}
BOUND_SIZE_X = BOUNDARY["maxX"] - BOUNDARY["minX"]
BOUND_SIZE_Y = BOUNDARY["maxY"] - BOUNDARY["minY"]
BEV_HEIGHT = 608  # rows  <- x (0..50m)
BEV_WIDTH = 608  # cols  <- y (-25..25m)
DISCRETIZATION = (BOUNDARY["maxX"] - BOUNDARY["minX"]) / BEV_HEIGHT


def _filter_lidar(lidar: np.ndarray) -> np.ndarray:
    """Keep points inside the BEV boundary and shift z to start at 0."""
    b = BOUNDARY
    mask = (
        (lidar[:, 0] >= b["minX"])
        & (lidar[:, 0] <= b["maxX"])
        & (lidar[:, 1] >= b["minY"])
        & (lidar[:, 1] <= b["maxY"])
        & (lidar[:, 2] >= b["minZ"])
        & (lidar[:, 2] <= b["maxZ"])
    )
    lidar = lidar[mask].copy()
    lidar[:, 2] = lidar[:, 2] - b["minZ"]
    return lidar


def _make_bev_map(lidar: np.ndarray) -> np.ndarray:
    """SFA3D makeBEVMap -> HWC uint8 [intensity, height, density] in [0,255]."""
    height, width = BEV_HEIGHT + 1, BEV_WIDTH + 1
    pc = np.copy(lidar)
    pc[:, 0] = np.int_(np.floor(pc[:, 0] / DISCRETIZATION))
    pc[:, 1] = np.int_(np.floor(pc[:, 1] / DISCRETIZATION) + width / 2)

    # Keep the top (highest-z) point per cell.
    order = np.lexsort((-pc[:, 2], pc[:, 1], pc[:, 0]))
    pc = pc[order]
    _, unique_idx, counts = np.unique(pc[:, 0:2], axis=0, return_index=True, return_counts=True)
    top = pc[unique_idx]

    height_map = np.zeros((height, width), dtype=np.float32)
    intensity_map = np.zeros((height, width), dtype=np.float32)
    density_map = np.zeros((height, width), dtype=np.float32)

    max_height = float(abs(BOUNDARY["maxZ"] - BOUNDARY["minZ"]))
    xi = np.int_(top[:, 0])
    yi = np.int_(top[:, 1])
    height_map[xi, yi] = top[:, 2] / max_height
    intensity_map[xi, yi] = top[:, 3]
    density_map[xi, yi] = np.minimum(1.0, np.log(counts + 1) / np.log(64))

    bev = np.zeros((BEV_HEIGHT, BEV_WIDTH, 3), dtype=np.float32)
    bev[:, :, 0] = intensity_map[:BEV_HEIGHT, :BEV_WIDTH]
    bev[:, :, 1] = height_map[:BEV_HEIGHT, :BEV_WIDTH]
    bev[:, :, 2] = density_map[:BEV_HEIGHT, :BEV_WIDTH]
    # ponytail: round-trips through uint8 to reuse the generic div-255 pipeline;
    # 8-bit quantization of a [0,1] BEV is negligible vs the int8-compiled model.
    return (bev * 255.0).astype(np.uint8)


def _inverse_rigid_trans(tr: np.ndarray) -> np.ndarray:
    """Inverse of a 3x4 [R|t] rigid transform, returned as 3x4."""
    inv = np.zeros_like(tr)
    inv[0:3, 0:3] = tr[0:3, 0:3].T
    inv[0:3, 3] = -tr[0:3, 0:3].T @ tr[0:3, 3]
    return inv


class _Calib:
    """Parse KITTI calib: V2C (Tr_velo_to_cam) and R0_rect."""

    def __init__(self, path: str) -> None:
        data = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                key, val = line.split(":", 1)
                data[key] = np.array([float(x) for x in val.split()], dtype=np.float64)
        self.V2C = data["Tr_velo_to_cam"].reshape(3, 4)
        self.R0 = data["R0_rect"].reshape(3, 3)
        r0 = np.eye(4)
        r0[:3, :3] = self.R0
        self.R0_inv = np.linalg.inv(r0)
        self.V2C_inv = _inverse_rigid_trans(self.V2C)

    def camera_to_lidar(self, x: float, y: float, z: float) -> np.ndarray:
        p = np.array([x, y, z, 1.0])
        p = self.R0_inv @ p
        p = self.V2C_inv @ p
        return p[0:3]


@DATASET_REGISTRY.register
class KITTI(DatasetBase):
    """KITTI 3D object detection — BEV input + LiDAR-frame GT boxes."""

    def __init__(self, data_dir: str, split: str = "val") -> None:
        self.ensure_exists(data_dir, _INSTALL_GUIDE)
        super().__init__(data_dir)
        self.split = split
        base = os.path.join(data_dir, "training")
        self.lidar_dir = os.path.join(base, "velodyne")
        self.label_dir = os.path.join(base, "label_2")
        self.calib_dir = os.path.join(base, "calib")

        split_file = os.path.join(data_dir, "ImageSets", f"{split}.txt")
        if not os.path.isfile(split_file):
            raise FileNotFoundError(f"KITTI split list not found: {split_file}\n{_INSTALL_GUIDE}")
        with open(split_file) as f:
            self.ids = [line.strip() for line in f if line.strip()]
        self.ids = [i for i in self.ids if os.path.exists(os.path.join(self.lidar_dir, f"{i}.bin"))]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Tuple, str]:
        sample_id = self.ids[idx]
        lidar = np.fromfile(os.path.join(self.lidar_dir, f"{sample_id}.bin"), dtype=np.float32)
        lidar = lidar.reshape(-1, 4)
        bev = _make_bev_map(_filter_lidar(lidar))
        image = self.preprocessing(bev)
        return image, bev.shape, sample_id

    def get_gt(self, sample_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return GT boxes in LiDAR coords and class ids for a frame.

        Boxes: [N, 7] = (x, y, z, h, w, l, yaw_lidar), restricted to the BEV
        boundary and the 3 evaluated classes.
        """
        label_path = os.path.join(self.label_dir, f"{sample_id}.txt")
        calib = _Calib(os.path.join(self.calib_dir, f"{sample_id}.txt"))
        boxes, classes = [], []
        if not os.path.exists(label_path):
            return np.zeros((0, 7), np.float64), np.zeros(0, np.float64)
        b = BOUNDARY
        with open(label_path) as f:
            for line in f:
                p = line.strip().split()
                if len(p) < 15:
                    continue
                cls_id = CLASS_NAME_TO_ID.get(p[0])
                if cls_id is None:
                    continue
                h, w, length = float(p[8]), float(p[9]), float(p[10])
                cx, cy, cz = float(p[11]), float(p[12]), float(p[13])
                ry = float(p[14])
                x, y, z = calib.camera_to_lidar(cx, cy, cz)
                if not (b["minX"] <= x < b["maxX"] and b["minY"] <= y < b["maxY"] and b["minZ"] <= z < b["maxZ"]):
                    continue
                if h <= 0 or w <= 0 or length <= 0:
                    continue
                yaw_lidar = -ry - np.pi / 2
                boxes.append([x, y, z, h, w, length, yaw_lidar])
                classes.append(cls_id)
        if not boxes:
            return np.zeros((0, 7), np.float64), np.zeros(0, np.float64)
        return np.array(boxes, np.float64), np.array(classes, np.float64)
