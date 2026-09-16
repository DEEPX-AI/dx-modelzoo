"""Tests for raw-layout Distinctions-646 composition."""

import cv2
import numpy as np

from dx_modelzoo.dataset.distinctions646 import Distinctions646
from dx_modelzoo.preprocessing import PreprocessingPipeline


def test_raw_layout_composes_without_val_directory(tmp_path):
    test_dir = tmp_path / "Test"
    (test_dir / "FG").mkdir(parents=True)
    (test_dir / "GT").mkdir()
    bg_dir = tmp_path / "backgrounds"
    bg_dir.mkdir()

    fg = np.full((2, 3, 3), 100, dtype=np.uint8)
    alpha = np.full((2, 3), 255, dtype=np.uint8)
    bg = np.full((2, 3, 3), 200, dtype=np.uint8)
    cv2.imwrite(str(test_dir / "FG" / "test_0.png"), fg)
    cv2.imwrite(str(test_dir / "GT" / "test_0.png"), alpha)
    cv2.imwrite(str(bg_dir / "bg_0.jpg"), bg)
    cv2.imwrite(str(bg_dir / "bg_1.jpg"), bg)
    (test_dir / "fg_test.txt").write_text("test_0.png\n")
    (test_dir / "bg_test.txt").write_text("bg_0.jpg\nbg_1.jpg\n")

    dataset = Distinctions646(str(tmp_path), background_dir=str(bg_dir))
    dataset.preprocessing = PreprocessingPipeline([{"type": "div", "x": 255}])

    image, target = dataset[0]
    assert len(dataset) == 2
    assert image.shape == (2, 3, 3)
    assert np.allclose(image, 100 / 255)
    assert np.allclose(target, 1)
