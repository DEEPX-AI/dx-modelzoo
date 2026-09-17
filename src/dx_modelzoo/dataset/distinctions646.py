from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.dataset import DATASET_REGISTRY

_INSTALL_GUIDE = """\
  [Distinctions-646] — image matting test set

  1. Obtain the official Distinctions-646 test data through the PaddleSeg
     PP-Matting release:
     https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.8/Matting
  2. Obtain the PASCAL VOC 2012 JPEGImages directory separately:
     http://host.robots.ox.ac.uk/pascal/VOC/voc2012/
  3. Arrange the files as follows:
       <DATA_ROOT>/Distinctions-646/
         Test/
           FG/
             ... foreground images ...
           GT/
             ... alpha masks matching foreground filenames ...
           fg_test.txt
           bg_test.txt
       <DATA_ROOT>/PascalVOC/VOCdevkit/VOC2012/JPEGImages/
         ... PASCAL VOC background images ...
  4. Each line in fg_test.txt names one file under Test/FG.
     Each line in bg_test.txt names one VOC background file, either relative
     to the JPEGImages directory or as an absolute path. The official test
     split contains 50 foregrounds and 20 backgrounds per foreground.

  Composite samples are generated in memory; no derived val/ directory is
  required. Follow the Distinctions-646 and PASCAL VOC 2012 terms for use and
  redistribution.
"""


@DATASET_REGISTRY.register
class Distinctions646(DatasetBase):
    """Official Distinctions-646 test set with in-memory composition.

    The raw release contains 50 foreground/alpha pairs and a 1000-entry
    foreground-background pairing list. Composite images are created per item
    so no derived ``val/`` directory is required.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "val",
        background_dir: str = "/mnt/datasets/PascalVOC/VOCdevkit/VOC2012/JPEGImages",
    ) -> None:
        self.ensure_exists(data_dir, _INSTALL_GUIDE)
        self.background_dir = Path(background_dir)
        super().__init__(data_dir)

        test_dir = Path(data_dir, "Test")
        fg_files = self._read_list(test_dir / "fg_test.txt")
        bg_files = self._read_list(test_dir / "bg_test.txt")
        if not fg_files or not bg_files:
            raise ValueError("Distinctions-646 fg_test.txt and bg_test.txt must be non-empty.")
        if len(bg_files) % len(fg_files):
            raise ValueError("Distinctions-646 background count must divide foreground count.")

        backgrounds_per_fg = len(bg_files) // len(fg_files)
        self.samples = []
        for index, fg_name in enumerate(fg_files):
            fg_path = test_dir / "FG" / fg_name
            alpha_path = test_dir / "GT" / fg_name
            for bg_name in bg_files[index * backgrounds_per_fg : (index + 1) * backgrounds_per_fg]:
                bg_path = Path(bg_name)
                if not bg_path.is_absolute():
                    bg_path = self.background_dir / bg_path
                self.samples.append((fg_path, alpha_path, bg_path))

        if not self.samples:
            raise ValueError(f"Distinctions-646 {split} split is empty.")

    @staticmethod
    def _read_list(path: Path) -> list[str]:
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def __len__(self) -> int:
        return len(self.samples)

    def validate(self) -> None:
        test_dir = Path(self.data_dir, "Test")
        required = ("FG", "GT", "fg_test.txt", "bg_test.txt")
        missing = [item for item in required if not (test_dir / item).exists()]
        if missing:
            raise RuntimeError(f"Missing Distinctions-646 entries under {test_dir}: {missing}")
        if not self.background_dir.is_dir():
            raise RuntimeError(f"Background directory not found: {self.background_dir}")

    def __getitem__(self, index: int) -> Tuple:
        fg_path, alpha_path, bg_path = self.samples[index]
        fg = cv2.imread(str(fg_path))
        alpha = cv2.imread(str(alpha_path), cv2.IMREAD_GRAYSCALE)
        bg = cv2.imread(str(bg_path))
        if fg is None:
            raise FileNotFoundError(f"Failed to load foreground: {fg_path}")
        if alpha is None:
            raise FileNotFoundError(f"Failed to load alpha: {alpha_path}")
        if bg is None:
            raise FileNotFoundError(f"Failed to load background: {bg_path}")
        if alpha.shape != fg.shape[:2]:
            raise ValueError(f"Alpha shape {alpha.shape} does not match foreground {fg.shape[:2]}: {fg_path}")

        height, width = fg.shape[:2]
        bg_height, bg_width = bg.shape[:2]
        ratio = max(height / bg_height, width / bg_width)
        if ratio > 1:
            pil_bg = Image.fromarray(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB))
            pil_bg = pil_bg.resize(
                (math.ceil(bg_width * ratio), math.ceil(bg_height * ratio)),
                Image.Resampling.BICUBIC,
            )
            bg = cv2.cvtColor(np.asarray(pil_bg), cv2.COLOR_RGB2BGR)

        bg = bg[:height, :width].astype(np.float64)
        alpha_float = (alpha.astype(np.float64) / 255.0)[:, :, None]
        composite = (alpha_float * fg.astype(np.float64) + (1 - alpha_float) * bg).astype(np.uint8)
        return self.preprocessing(composite), alpha.astype(np.float32) / 255.0
