"""Tests for CamVid split validation."""

import cv2
import numpy as np

from dx_modelzoo.dataset.camvid import CamVid
from dx_modelzoo.preprocessing import PreprocessingPipeline


def test_camvid_validates_split_and_loads_sample(tmp_path):
    images = tmp_path / "images"
    annot = tmp_path / "annot"
    images.mkdir()
    annot.mkdir()
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    label = np.zeros((2, 3), dtype=np.uint8)
    cv2.imwrite(str(images / "sample.png"), image)
    cv2.imwrite(str(annot / "sample.png"), label)
    (tmp_path / "test.txt").write_text("images/sample.png annot/sample.png\n")

    dataset = CamVid(str(tmp_path), split="test")
    dataset.preprocessing = PreprocessingPipeline([{"type": "div", "x": 255}])

    result, target = dataset[0]
    assert len(dataset) == 1
    assert result.shape == (2, 3, 3)
    assert target.shape == (2, 3)
