from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import cv2
import numpy as np
from cv2 import INTER_AREA, INTER_LINEAR
from cv2 import resize as cv2_resize
from PIL import Image

from dx_modelzoo.preprocessing import PREPROCESSING_REGISTRY
from dx_modelzoo.preprocessing.enums import (
    AlignSideEnum,
    BackendEnum,
    CVResizeInterpolationEnum,
    InterpolationEnum,
    PILResizeInterpolationEnum,
    ResizeArgEnum,
    ResizeMode,
)

EPSILON = 1e-10

ARGS_FOR_MODE = {
    ResizeMode.default: {
        ResizeArgEnum.backend: BackendEnum.cv2,
        ResizeArgEnum.align_side: AlignSideEnum.both,
        ResizeArgEnum.scale_method: None,
        ResizeArgEnum.interpolation: InterpolationEnum.LINEAR,
    },
    ResizeMode.torchvision: {
        ResizeArgEnum.backend: BackendEnum.pil,
        ResizeArgEnum.align_side: AlignSideEnum.short,
        ResizeArgEnum.scale_method: None,
        ResizeArgEnum.interpolation: InterpolationEnum.BILINEAR,
    },
    ResizeMode.pad: {
        ResizeArgEnum.backend: BackendEnum.cv2,
        ResizeArgEnum.align_side: AlignSideEnum.long,
        ResizeArgEnum.scale_method: None,
        ResizeArgEnum.interpolation: InterpolationEnum.LINEAR,
    },
    ResizeMode.pycls: {
        ResizeArgEnum.backend: BackendEnum.cv2,
        ResizeArgEnum.align_side: AlignSideEnum.short,
        ResizeArgEnum.scale_method: None,
        ResizeArgEnum.interpolation: None,
    },
}


@dataclass
class ResizeArgs:
    backend: BackendEnum
    size: Optional[Union[int, List]] = None
    interpolation: InterpolationEnum = InterpolationEnum.LINEAR
    width: Optional[int] = None
    height: Optional[int] = None
    align_side: Optional[AlignSideEnum] = None
    scale_method: Optional[str] = None
    pad_location: Optional[str] = None
    pad_value: Optional[List[int]] = None

    def __post_init__(self) -> None:
        if isinstance(self.size, int):
            self.size = [self.size, self.size]
        if self.size is None and self.width is not None and self.height is not None:
            self.size = [self.height, self.width]


class CV2Resize:
    def __init__(self, size, interpolation, *args, **kwargs):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, inputs, aligned_size, ratios=None, *args, **kwargs):
        if isinstance(inputs, Image.Image):
            inputs = np.array(inputs)
        h, w = aligned_size
        if self.interpolation is None:
            hr, wr = ratios or (1, 1)
            interp = INTER_LINEAR if hr > 1 else INTER_AREA
        else:
            interp = CVResizeInterpolationEnum[self.interpolation]
        return cv2_resize(inputs, (w, h), interpolation=interp)


class TorchVisionResize:
    def __init__(self, size, interpolation, *args, **kwargs):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, inputs, aligned_size, *args, **kwargs):
        if isinstance(inputs, np.ndarray):
            inputs = Image.fromarray(inputs)
        h, w = aligned_size
        return np.array(inputs.resize((w, h), PILResizeInterpolationEnum[self.interpolation]))


class PadResize:
    def __init__(self, size, interpolation, pad_location, pad_value, *args, **kwargs):
        self.height, self.width = size
        self.interpolation = CVResizeInterpolationEnum[interpolation]
        self.pad_location = pad_location
        self.pad_value = pad_value

    def __call__(self, inputs, aligned_size, ratios=None, *args, **kwargs):
        if isinstance(inputs, Image.Image):
            inputs = np.array(inputs)
        h, w = aligned_size
        if ratios:
            hr, wr = ratios
            interp = self.interpolation
            if hr == wr and hr != 1:
                interp = INTER_LINEAR if hr > 1 else INTER_AREA
        else:
            interp = self.interpolation
        resized = cv2.resize(inputs, (w, h), interpolation=interp)
        return self._pad(np.array(resized), (h, w))

    def _pad(self, inputs, aligned_size):
        hp = self.height - aligned_size[0]
        wp = self.width - aligned_size[1]
        if self.pad_location == "edge":
            hp /= 2
            wp /= 2
            top, bottom = int(round(hp - 0.1)), int(round(hp + 0.1))
            left, right = int(round(wp - 0.1)), int(round(wp + 0.1))
        elif self.pad_location == "back":
            top, left = 0, 0
            bottom, right = int(round(hp)), int(round(wp))
        else:
            raise ValueError(f"Invalid pad_location: {self.pad_location}")
        return cv2.copyMakeBorder(inputs, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.pad_value)


@PREPROCESSING_REGISTRY.register("resize")
class Resize:
    """Resize the input to the given size."""

    def __init__(self, mode=None, **args):
        if mode is None:
            mode = ResizeMode.default
        if not ResizeMode.has_value(mode):
            raise ValueError(f"Invalid mode: {mode}")
        self.resize_args = ResizeArgs(**self._get_default_args(mode, args))
        self.resize_method = self._get_resize_method(mode)

    def _get_default_args(self, mode, args):
        defaults = ARGS_FOR_MODE[mode]
        for key in [
            ResizeArgEnum.backend,
            ResizeArgEnum.align_side,
            ResizeArgEnum.scale_method,
            ResizeArgEnum.interpolation,
        ]:
            if args.get(key) is None:
                args[key] = defaults[key]
        return args

    def _get_resize_method(self, mode) -> Callable:
        if mode in (ResizeMode.default, ResizeMode.pycls):
            return CV2Resize(**self.resize_args.__dict__)
        elif mode == ResizeMode.torchvision:
            return TorchVisionResize(**self.resize_args.__dict__)
        elif mode == ResizeMode.pad:
            return PadResize(**self.resize_args.__dict__)
        raise ValueError(f"Invalid mode: {mode}")

    def _align_size(self, image_size):
        h, w = image_size

        # Float ratio mode: scale both dimensions by the ratio
        if isinstance(self.resize_args.size, float):
            ratio = self.resize_args.size
            ah = int(h * ratio + EPSILON)
            aw = int(w * ratio + EPSILON)
            return (ah, aw), (ratio, ratio)

        short = min(h, w)
        align = self.resize_args.align_side
        if align == AlignSideEnum.short:
            rv = (short, short)
        elif align == AlignSideEnum.long:
            # Letterbox: scale uniformly by the smaller ratio so the image fits
            # inside the (possibly rectangular) target, padding the remainder.
            ratio = min(self.resize_args.size[0] / h, self.resize_args.size[1] / w)
            ah = int(ratio * h + EPSILON)
            aw = int(ratio * w + EPSILON)
            return (ah, aw), (ratio, ratio)
        elif align == AlignSideEnum.both:
            rv = (h, w)
        else:
            raise ValueError(f"Invalid align_side: {align}")
        hr = self.resize_args.size[0] / rv[0]
        wr = self.resize_args.size[1] / rv[1]
        ah = int(hr * h + EPSILON)
        aw = int(wr * w + EPSILON)
        return (ah, aw), (hr, wr)

    def __call__(self, inputs):
        if isinstance(inputs, Image.Image):
            image_size = (inputs.size[1], inputs.size[0])
        else:
            image_size = inputs.shape[:2]
        aligned_size, ratios = self._align_size(image_size)
        return self.resize_method(inputs, aligned_size, ratios)
