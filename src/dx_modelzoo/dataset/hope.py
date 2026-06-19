"""HOPE-Image dataset for single-object 6-DoF pose evaluation (NVIDIA Isaac DOPE).

Reads the HOPE-Image ``valid`` split for one target object class (DOPE is
trained per-object) and yields the projected 3D centroid as 2D ground truth in
normalised image coordinates.
"""

from __future__ import annotations

import json
import os
from glob import glob
from typing import Dict, List, Tuple

import cv2
import numpy as np

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.dataset import DATASET_REGISTRY

# --------------------------------------------------------------------------- #
# HOPE / DOPE cuboid geometry.                                                 #
#                                                                              #
# Two distinct facts kept together for the single-object DOPE pipeline:        #
#   * HOPE object dimensions (cm) and rotational-symmetry flags (dataset).     #
#   * The DOPE cuboid vertex ordering (the 9 belief-map channel order) and the #
#     fixed signed-axis permutation mapping the DOPE cuboid frame to the       #
#     HOPE/BOP object-model frame (model/training convention).                 #
# Used here to project GT cuboid keypoints into the DOPE channel order, and by #
# the DOPE custom_ops (PnP) via the dataset attributes exposed below.          #
# --------------------------------------------------------------------------- #

# Published HOPE cuboid dimensions (cm, x/y/z).
# Source: NVlabs/Deep_Object_Pose config/config_pose.yaml ("Cuboid dimension in
# cm x,y,z"). Units match the HOPE GT pose translations (cm), so PnP/ADD are cm.
HOPE_CUBOID_DIMS: Dict[str, Tuple[float, float, float]] = {
    # original DOPE / YCB-style objects
    "cracker": (16.4036006927, 21.3437004089, 7.1799998283),
    "gelatin": (8.9182996750, 7.3115000725, 2.9983000755),
    "meat": (10.1646738052, 8.3542995453, 5.7600898743),
    "mustard": (9.6024150848, 19.1301002502, 5.8248949051),
    "soup": (6.7659378052, 10.1855001450, 6.7714257240),
    "sugar": (9.2677307129, 17.6253395081, 4.5134143829),
    "bleach": (10.2677307129, 26.6253395081, 7.5134143829),
    # HOPE objects
    "AlphabetSoup": (8.3555002213, 7.1121001244, 6.6055998802),
    "Butter": (5.2825999260, 2.3935999870, 10.3301000595),
    "Ketchup": (14.8607997894, 4.3368000984, 6.4513998032),
    "Pineapple": (5.7623000145, 6.9598999023, 6.5675001144),
    "BBQSauce": (14.8329000473, 4.3478999138, 6.4632000923),
    "MacaroniAndCheese": (16.6256008148, 4.0180997849, 12.3508996964),
    "Popcorn": (8.4976997375, 3.8252000809, 12.6492004395),
    "Mayo": (14.7902002335, 4.1030998230, 6.4541001320),
    "Raisins": (12.3175001144, 3.9751999378, 8.5874996185),
    "Cherries": (5.8038997650, 7.0907998085, 6.6101999283),
    "Milk": (19.0358009338, 7.3262000084, 7.2154998779),
    "SaladDressing": (14.7440996170, 4.3695998192, 6.4039001465),
    "ChocolatePudding": (4.9471998215, 2.9923000336, 8.3498001099),
    "Mushrooms": (3.3322000504, 7.0798997879, 6.5869998932),
    "Spaghetti": (4.9836997986, 2.8492999077, 24.9881000519),
    "Cookies": (16.7243003845, 4.0152001381, 12.2746000290),
    "Mustard": (16.0049991608, 4.8573999405, 6.5132999420),
    "TomatoSauce": (8.2847003937, 7.0198001862, 6.6469998360),
    "Corn": (5.8038997650, 7.0907998085, 6.6101999283),
    "OrangeJuice": (19.2483005524, 7.2781000137, 7.1582999229),
    "Tuna": (3.2571001053, 7.0805997849, 6.5837001801),
    "CreamCheese": (5.3206000328, 2.4230999947, 10.3590002060),
    "Parmesan": (10.2861995697, 6.6093001366, 7.1117000580),
    "Yogurt": (5.3677000999, 6.7961997986, 6.7915000916),
    "GranolaBars": (12.4006004333, 3.8738000393, 16.5338001251),
    "Peaches": (5.7781000137, 7.0961999893, 6.5925998688),
    "GreenBeans": (5.7586998940, 7.0608000755, 6.5732002258),
    "PeasAndCarrots": (5.8512001038, 7.0636000633, 6.5918002129),
}

# Rotationally-symmetric HOPE objects (scored with ADD-S instead of ADD).
HOPE_SYMMETRIC_OBJECTS = frozenset(
    {
        "soup",
        "meat",
        "AlphabetSoup",
        "Pineapple",
        "Cherries",
        "Mushrooms",
        "TomatoSauce",
        "Corn",
        "Tuna",
        "Parmesan",
        "Yogurt",
        "Peaches",
        "GreenBeans",
        "PeasAndCarrots",
    }
)

# DOPE cuboid frame -> HOPE/BOP object-model frame alignment (signed-axis
# permutation). A pose recovered by PnP on the DOPE cuboid is rotated w.r.t. the
# GT pose by this constant; used as R_aligned = R_gt @ DOPE_TO_HOPE_FRAME.
DOPE_TO_HOPE_FRAME = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)


def dope_cuboid_vertices(dims: Tuple[float, float, float]) -> np.ndarray:
    """Return the 9 DOPE cuboid vertices (cm) in ``CuboidVertexType`` order.

    Matches NVlabs/Deep_Object_Pose ``Cuboid3d.generate_vertexes`` (OpenCV frame,
    origin at cuboid centre): X->right, Y->down, Z->forward. Index order matches
    the 9 belief-map channels: FrontTopRight(0), FrontTopLeft(1),
    FrontBottomLeft(2), FrontBottomRight(3), RearTopRight(4), RearTopLeft(5),
    RearBottomLeft(6), RearBottomRight(7), Center(8).
    """
    w, h, d = float(dims[0]), float(dims[1]), float(dims[2])
    right, left = w / 2.0, -w / 2.0
    top, bottom = -h / 2.0, h / 2.0
    front, rear = d / 2.0, -d / 2.0
    return np.array(
        [
            [right, top, front],  # 0 Front Top Right
            [left, top, front],  # 1 Front Top Left
            [left, bottom, front],  # 2 Front Bottom Left
            [right, bottom, front],  # 3 Front Bottom Right
            [right, top, rear],  # 4 Rear Top Right
            [left, top, rear],  # 5 Rear Top Left
            [left, bottom, rear],  # 6 Rear Bottom Left
            [right, bottom, rear],  # 7 Rear Bottom Right
            [0.0, 0.0, 0.0],  # 8 Center
        ],
        dtype=np.float64,
    )


def project_cuboid_keypoints(
    pose: np.ndarray,
    cuboid3d: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    frame_align: np.ndarray = DOPE_TO_HOPE_FRAME,
) -> np.ndarray:
    """Project the 9 cuboid vertices into normalised [0,1] image coordinates.

    The GT rotation is expressed in the DOPE cuboid frame
    (``R = R_gt @ frame_align``) so the projected 2D points follow the DOPE
    belief-map channel order and are directly comparable to the model output.
    """
    pose = np.asarray(pose, dtype=np.float64).reshape(4, 4)
    R = pose[:3, :3] @ frame_align
    t = pose[:3, 3]
    pts_cam = cuboid3d @ R.T + t  # (9, 3) camera frame (cm)
    proj = (K @ pts_cam.T).T  # (9, 3)
    xy = proj[:, :2] / proj[:, 2:3]
    xy[:, 0] /= float(width)
    xy[:, 1] /= float(height)
    return xy


_INSTALL_GUIDE = (
    "  [HOPE] — HOPE-Image dataset (NVIDIA Isaac DOPE 6-DoF pose, research use)\n\n"
    "  Expected layout under <DATA_ROOT> (point eval_path / --data-root here):\n"
    "      <root>/\n"
    "        valid/\n"
    "          scene_0000/\n"
    "            0000_rgb.jpg\n"
    "            0000_depth.png\n"
    "            0000.json        per-frame 6-DoF GT (objects[].class + 4x4 pose, camera.intrinsics)\n"
    "            0001_rgb.jpg ...\n"
    "          scene_0001/ ...\n"
    "        test/   (NO ground-truth poses — do NOT use for accuracy)\n\n"
    "  Source : https://github.com/swtyree/hope-dataset  (HOPE-Image release)\n"
    "  License: research use only.\n"
)


@DATASET_REGISTRY.register
class HOPE(DatasetBase):
    """Single-object HOPE-Image subset for DOPE 6-DoF pose evaluation.

    Args:
        data_dir: HOPE root containing the ``<split>/scene_XXXX`` tree.
        object_class: target HOPE object name (e.g. ``"BBQSauce"``) — DOPE is per-object.
        split: dataset split, ``"valid"`` (default; ``"test"`` has no GT).
        max_samples: cap on number of indexed frames (0 = all).
    """

    def __init__(
        self,
        data_dir: str,
        object_class: str = "BBQSauce",
        split: str = "valid",
        max_samples: int = 0,
    ) -> None:
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            if os.path.isdir(os.path.join(data_dir, "scene_0000")):
                split_dir = data_dir
        self.ensure_exists(split_dir, _INSTALL_GUIDE)
        super().__init__(data_dir)
        self.object_class = object_class
        self.split = split
        self.split_dir = split_dir
        self.max_samples = int(max_samples) if max_samples else 0

        # DOPE cuboid geometry for this object (None if dimensions unknown):
        # enables projected GT keypoints (2D) and PnP-based ADD (3D).
        dims = HOPE_CUBOID_DIMS.get(object_class)
        if dims is not None:
            self.cuboid3d = dope_cuboid_vertices(dims)  # (9, 3) cm, DOPE order
            self.diameter = float(np.linalg.norm(np.array(dims, dtype=np.float64)))
            self.is_symmetric = object_class in HOPE_SYMMETRIC_OBJECTS
            self.frame_align = DOPE_TO_HOPE_FRAME
        else:
            self.cuboid3d = None
            self.diameter = 0.0
            self.is_symmetric = False
            self.frame_align = DOPE_TO_HOPE_FRAME

        self.index: List[Tuple[str, str]] = []
        self._build_index()

    def _build_index(self) -> None:
        jsons = sorted(glob(os.path.join(self.split_dir, "scene_*", "[0-9]*.json")))
        skipped = 0
        for jpath in jsons:
            try:
                gt = self._load_gt(jpath)
            except (KeyError, ValueError, TypeError, json.JSONDecodeError):
                skipped += 1
                continue
            if gt["num"] <= 0:
                continue
            rgb = jpath[:-5] + "_rgb.jpg"
            if not os.path.isfile(rgb):
                continue
            self.index.append((rgb, jpath))
            if self.max_samples and len(self.index) >= self.max_samples:
                return
        if jsons and not self.index:
            raise RuntimeError(
                f"HOPE index is empty: parsed {len(jsons)} GT json file(s) under "
                f"{self.split_dir} but none yielded a '{self.object_class}' instance "
                f"({skipped} failed to parse). Check object_class / split / dataset layout."
            )

    def _load_gt(self, jpath: str) -> Dict:
        with open(jpath) as f:
            d = json.load(f)
        cam = d["camera"]
        K = np.array(cam["intrinsics"], dtype=np.float64).reshape(3, 3)
        W = int(cam["width"])
        H = int(cam["height"])
        poses: List = []
        centroids: List = []
        translations: List = []
        keypoints_2d: List = []
        for obj in d["objects"]:
            if obj.get("class") != self.object_class:
                continue
            pose = np.array(obj["pose"], dtype=np.float64).reshape(4, 4)
            t = pose[:3, 3]
            if t[2] <= 0:
                continue
            p = K @ (t / t[2])
            centroids.append(np.array([p[0] / W, p[1] / H], dtype=np.float64))
            poses.append(pose)
            translations.append(t.copy())
            if self.cuboid3d is not None:
                # Projected pseudo-GT cuboid keypoints (8 corners + centroid) in
                # normalised image coords, in the DOPE belief-map channel order.
                keypoints_2d.append(
                    project_cuboid_keypoints(
                        pose, self.cuboid3d, K, W, H, self.frame_align
                    )
                )
        n = len(poses)
        return {
            "class": self.object_class,
            "num": n,
            "K": K,
            "W": W,
            "H": H,
            "poses": np.array(poses, dtype=np.float64).reshape(n, 4, 4)
            if n
            else np.zeros((0, 4, 4)),
            "translations": np.array(translations, dtype=np.float64).reshape(n, 3)
            if n
            else np.zeros((0, 3)),
            "centroids_2d": np.array(centroids, dtype=np.float64).reshape(n, 2)
            if n
            else np.zeros((0, 2)),
            "keypoints_2d": (
                np.array(keypoints_2d, dtype=np.float64).reshape(n, 9, 2)
                if (n and keypoints_2d)
                else np.zeros((0, 9, 2))
            ),
        }

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Tuple:
        rgb_path, jpath = self.index[idx]
        img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read HOPE RGB frame: {rgb_path}")
        gt = self._load_gt(jpath)
        meta = {
            "object_class": self.object_class,
            "rgb_path": rgb_path,
            "height": gt["H"],
            "width": gt["W"],
            "index": idx,
        }
        image = self.preprocessing(img)
        return image, gt, meta
