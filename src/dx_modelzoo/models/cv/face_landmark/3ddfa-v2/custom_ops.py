"""TDDFA_V2 3DMM parameter decoder.

Decodes 62-dimensional 3DMM parameters into 68 2D face landmarks
using BFM basis matrices. Follows the reference BFMDecoder implementation.
Parameters layout: [12 pose | 40 shape | 10 expression].
"""
from __future__ import annotations

import os
import pickle

import numpy as np

from dx_modelzoo.postprocessing import POSTPROCESSING_REGISTRY


@POSTPROCESSING_REGISTRY.register("tddfa_decode")
class TDDFADecode:
    """Decode TDDFA_V2 output (1, 62) to 68 2D landmarks.

    Args:
        bfm_dir: Directory containing bfm_noneck_v3.pkl and param_mean_std_62d_120x120.pkl.
        size: Crop size used for Y-flip (default 120).
    """

    def __init__(self, bfm_dir: str, size: int = 120, **kwargs) -> None:
        bfm_path = os.path.join(bfm_dir, "bfm_noneck_v3.pkl")
        param_path = os.path.join(bfm_dir, "param_mean_std_62d_120x120.pkl")

        with open(bfm_path, "rb") as f:
            bfm = pickle.load(f, encoding="latin1")
        with open(param_path, "rb") as f:
            param_mean_std = pickle.load(f, encoding="latin1")

        self._param_mean = param_mean_std["mean"]
        self._param_std = param_mean_std["std"]

        keypoints = bfm["keypoints"].astype(np.int64)
        self._u = bfm["u"][keypoints].reshape(-1, 1).astype(np.float32)
        self._w_shp = bfm["w_shp"][keypoints].astype(np.float32)[:, :40]
        self._w_exp = bfm["w_exp"][keypoints].astype(np.float32)[:, :10]
        self._size = size

    def __call__(self, outputs, **kwargs) -> np.ndarray:
        params = np.asarray(outputs).flatten().astype(np.float64)

        # Denormalize
        params = params * self._param_std + self._param_mean

        R_ = params[:12].reshape(3, -1)
        R = R_[:, :3]
        offset = R_[:, -1:].reshape(3, 1)
        alpha_shp = params[12:52].reshape(-1, 1)
        alpha_exp = params[52:62].reshape(-1, 1)

        vertex = self._u + self._w_shp @ alpha_shp + self._w_exp @ alpha_exp
        vertex = vertex.reshape(3, -1, order="F")  # [3, 68]

        pts = R @ vertex + offset  # [3, 68]

        # Convert to image convention (Y-flip)
        pts[0, :] -= 1
        pts[1, :] = self._size - pts[1, :]

        return pts[:2, :].T.astype(np.float32)  # [68, 2]
