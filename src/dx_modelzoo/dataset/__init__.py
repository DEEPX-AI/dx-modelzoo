from __future__ import annotations

from dx_modelzoo.common.registry import Registry

__all__ = ["DATASET_REGISTRY"]

DATASET_REGISTRY = Registry("dataset")  # Import submodules to trigger registration
from dx_modelzoo.dataset import ade20k  # noqa: F401, E402
from dx_modelzoo.dataset import aflw20003d  # noqa: F401, E402
from dx_modelzoo.dataset import bdd100k  # noqa: F401, E402
from dx_modelzoo.dataset import bsd  # noqa: F401, E402
from dx_modelzoo.dataset import celeba  # noqa: F401, E402
from dx_modelzoo.dataset import cityscapes  # noqa: F401, E402
from dx_modelzoo.dataset import coco  # noqa: F401, E402
from dx_modelzoo.dataset import coco_multiinput  # noqa: F401, E402
from dx_modelzoo.dataset import dotav1  # noqa: F401, E402
from dx_modelzoo.dataset import hand_keypoints  # noqa: F401, E402
from dx_modelzoo.dataset import hope  # noqa: F401, E402
from dx_modelzoo.dataset import hpatches  # noqa: F401, E402
from dx_modelzoo.dataset import imagenet  # noqa: F401, E402
from dx_modelzoo.dataset import kitti  # noqa: F401, E402
from dx_modelzoo.dataset import lfw  # noqa: F401, E402
from dx_modelzoo.dataset import lol  # noqa: F401, E402
from dx_modelzoo.dataset import market1501  # noqa: F401, E402
from dx_modelzoo.dataset import nyu  # noqa: F401, E402
from dx_modelzoo.dataset import objectron  # noqa: F401, E402
from dx_modelzoo.dataset import oxford_iiit_pet  # noqa: F401, E402
from dx_modelzoo.dataset import peta  # noqa: F401, E402
from dx_modelzoo.dataset import synthetic  # noqa: F401, E402
from dx_modelzoo.dataset import voc  # noqa: F401, E402
from dx_modelzoo.dataset import widerface  # noqa: F401, E402
