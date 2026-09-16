from __future__ import annotations

import os
from glob import glob
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import cv2
import numpy as np

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.dataset import DATASET_REGISTRY

_INSTALL_GUIDE = """\
  [ImageNet ILSVRC2012] — Academic/Research use only

  1. Register at https://image-net.org/download-images.php
  2. Download "ILSVRC2012_img_val.tar" (validation images, ~6.3GB)
  3. Download "ILSVRC2012_devkit_t12.tar.gz" (devkit)
  4. Extract to: <DATA_ROOT>/ILSVRC2012/val/
     Expected structure:
       ILSVRC2012/val/
         n01440764/
           ILSVRC2012_val_00000293.JPEG
           ...
         n01443537/
           ...
  5. Use the devkit script to organize images into per-class subdirectories.

  License: Research/academic use only. Commercial use is NOT permitted.
  See: https://image-net.org/download-images.php
"""


def _build_clip_text_inputs(
    model,
    tokenizer: Callable,
    classnames: Sequence[str],
    template: Callable[[str], str],
):
    """Build token-embedding streams consumed by an exported CLIP text transformer."""
    tokens = tokenizer([template(classname) for classname in classnames])
    return model.token_embedding(tokens) + model.positional_embedding


@DATASET_REGISTRY.register
class ILSVRC2012(DatasetBase):
    """ImageNet dataset. Reads images from class subdirectories.

    Expected structure:
        data_dir/
            class_a/
                xxxxx.JPEG
            class_b/
                xxxxx.JPEG
    """

    def __init__(self, data_dir: str) -> None:
        self.ensure_exists(data_dir, _INSTALL_GUIDE)
        super().__init__(data_dir)
        self.image_files = sorted(glob(os.path.join(self.data_dir, "**", "*.JPEG"), recursive=True))
        self._class_map = self._build_class_map()
        self.class_list = self._build_class_list()

    def _build_class_map(self) -> Dict[str, int]:
        dirs = sorted([d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))])
        return {name: idx for idx, name in enumerate(dirs)}

    def _build_class_list(self) -> List[int]:
        class_list = []
        for path in self.image_files:
            class_name = path.split(os.sep)[-2]
            class_list.append(self._class_map[class_name])
        return class_list

    @property
    def class_map(self) -> Dict[str, int]:
        return self._class_map

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        img = cv2.imread(self.image_files[idx])
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {self.image_files[idx]}")
        img = self.preprocessing(img)
        label = self.class_list[idx]
        return img, label

    def validate(self) -> None:
        """Validate the dataset's integrity and structure.

        Raises:
            RuntimeError: If the dataset is invalid or corrupted.
        """
        # directory structure
        is_correct_structure = Path(self.data_dir).is_dir()
        if not is_correct_structure:
            raise RuntimeError(
                "Dataset directory structure is invalid. Expected class subdirectories with JPEG images under"
                f" {self.data_dir}."
            )

        # file existence
        NUM_DATA = 50000
        NUM_CLASSES = 1000
        num_images = len(list(Path(self.data_dir).glob("*/*.JPEG")))
        if num_images != NUM_DATA:
            raise RuntimeError(f"Expected {NUM_DATA} images in {self.data_dir}, but found {num_images}.")

        num_classes = len(list(Path(self.data_dir).glob("*")))
        if num_classes != NUM_CLASSES:
            raise RuntimeError(f"Expected {NUM_CLASSES} classes in {self.data_dir}, but found {num_classes}.")


@DATASET_REGISTRY.register
class ImageNetText(DatasetBase):
    """ImageNet class prompts converted to CLIP text-transformer inputs."""

    def __init__(
        self,
        data_dir: str = ".",
        model_name: str = "RN50-quickgelu",
        pretrained: str = "openai",
    ) -> None:
        import open_clip
        import torch
        from open_clip.zero_shot_metadata import IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES

        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device="cpu")
        model.eval()
        tokenizer = open_clip.get_tokenizer(model_name)

        self.token_ids = tokenizer([OPENAI_IMAGENET_TEMPLATES[0](name) for name in IMAGENET_CLASSNAMES]).numpy()
        with torch.no_grad():
            prompts = _build_clip_text_inputs(
                model,
                tokenizer,
                IMAGENET_CLASSNAMES,
                OPENAI_IMAGENET_TEMPLATES[0],
            )
        self.prompts = prompts.cpu().numpy().astype(np.float32)
        super().__init__(data_dir or ".")

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        return self.prompts[idx], idx

    def validate(self) -> None:
        if self.prompts.ndim != 3 or self.prompts.shape[1:] != (77, 512):
            raise ValueError(f"Unexpected CLIP prompt shape: {self.prompts.shape}")
