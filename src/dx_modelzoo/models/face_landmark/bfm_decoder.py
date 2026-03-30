"""BFM (Basel Face Model) decoder for 3DDFA_V2.

Decodes 62-dimensional 3DMM parameters into 68 2D face landmarks.
"""

import os
import pickle

import numpy as np


class BFMDecoder:
    """Decode 3DMM parameters to 68 face landmarks using BFM model.

    Args:
        bfm_dir: Directory containing bfm_noneck_v3.pkl and param_mean_std_62d_120x120.pkl
    """

    def __init__(self, bfm_dir: str):
        bfm_path = os.path.join(bfm_dir, "bfm_noneck_v3.pkl")
        param_path = os.path.join(bfm_dir, "param_mean_std_62d_120x120.pkl")

        with open(bfm_path, "rb") as f:
            bfm = pickle.load(f, encoding="latin1")
        with open(param_path, "rb") as f:
            param_mean_std = pickle.load(f, encoding="latin1")

        self.param_mean = param_mean_std["mean"]
        self.param_std = param_mean_std["std"]

        keypoints = bfm["keypoints"].astype(np.int64)
        self.u_base = bfm["u"][keypoints].reshape(-1, 1).astype(np.float32)
        self.w_shp_base = bfm["w_shp"][keypoints].astype(np.float32)[:, :40]
        self.w_exp_base = bfm["w_exp"][keypoints].astype(np.float32)[:, :10]

    def __call__(self, params: np.ndarray, size: int = 120) -> np.ndarray:
        """Decode 62-dim 3DMM parameters to 68 2D landmarks in crop space.

        Args:
            params: [1, 62] or [62] array of 3DMM parameters (normalized)
            size: Crop size (default 120)

        Returns:
            [68, 2] array of 2D landmark coordinates in crop image space
        """
        params = params.flatten().astype(np.float64)

        # Denormalize
        params = params * self.param_std + self.param_mean

        # Parse: first 12 params form [3, 4] matrix (rotation + translation)
        R_ = params[:12].reshape(3, -1)
        R = R_[:, :3]
        offset = R_[:, -1:].reshape(3, 1)
        alpha_shp = params[12:52].reshape(-1, 1)
        alpha_exp = params[52:62].reshape(-1, 1)

        # Reconstruct 3D keypoints
        vertex = self.u_base + self.w_shp_base @ alpha_shp + self.w_exp_base @ alpha_exp
        vertex = vertex.reshape(3, -1, order="F")  # [3, 68]

        # Apply rotation and translation
        pts = R @ vertex + offset  # [3, 68]

        # Convert to image convention (Y-flip) for crop coordinate space
        pts[0, :] -= 1
        pts[1, :] = size - pts[1, :]

        return pts[:2, :].T.astype(np.float32)  # [68, 2]
