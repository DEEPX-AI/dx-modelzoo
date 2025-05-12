from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
from cv2 import INTER_AREA, INTER_LINEAR
from cv2 import resize as cv2_resize
from PIL import Image

from dx_modelzoo.preprocessing.enums import (
    AlignSideEnum,
    BackendEnum,
    CVResizeInterpolationEnum,
    InterpolationEnum,
    PILResizeInterpolationEnum,
    ResizeArgEnum,
    ResizeMode,
    ScaleMethodEnum,
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
    size: Optional[int | List] = None
    interpolation: InterpolationEnum = InterpolationEnum.LINEAR
    width: Optional[int] = None
    height: Optional[int] = None
    align_side: Optional[AlignSideEnum] = None
    scale_method: Optional[ScaleMethodEnum] = None
    pad_location: Optional[str] = None
    pad_value: Optional[List[int]] = None

    def __post_init__(self) -> None:
        if isinstance(self.size, int):
            self.size = [self.size, self.size]

        if self.size is None and self.width is not None and self.height is not None:
            self.size = [self.height, self.width]


class CV2Resize:
    """CV2 Resize."""

    def __init__(self, size: List[int], interpolation: InterpolationEnum, *args, **kwargs) -> None:
        self.size = size
        self.interpolation = interpolation

    def __call__(self, inputs: np.ndarray | Image.Image, aligned_size: Tuple[int, int], *args, **kwargs) -> np.ndarray:
        if isinstance(inputs, Image.Image):
            inputs = np.array(inputs)
        aligned_height, aligned_width = aligned_size
        return cv2_resize(inputs, (aligned_width, aligned_height), CVResizeInterpolationEnum[self.interpolation])


class TorchVisionResize:
    def __init__(self, size: List[int], interpolation: InterpolationEnum, *args, **kwargs) -> None:
        """Torchvision Resize.
        it use PIL.Image's resize. almost TorchVision models use it.
        """
        self.size = size
        self.interpolation = interpolation

    def __call__(
        self,
        inputs: np.ndarray | Image.Image,
        aligned_size: Tuple[int, int],
        *args,
        **kwargs,
    ) -> np.ndarray:
        if isinstance(inputs, np.ndarray):
            inputs = Image.fromarray(inputs)
        aligned_height, aligned_width = aligned_size
        return np.array(inputs.resize((aligned_width, aligned_height), PILResizeInterpolationEnum[self.interpolation]))


class PadResize:
    """Pad Resize.
    it performs like 'tf.image.resize_with_pad' using PIL Image's resize.
    """

    def __init__(
        self,
        size: int | List[int],
        interpolation: InterpolationEnum,
        pad_location: str,
        pad_value: List[int],
        *args,
        **kwargs,
    ) -> None:
        self.width, self.height = size
        self.interpolation = PILResizeInterpolationEnum[interpolation]
        self.pad_location = pad_location
        self.pad_value = pad_value

    def __call__(
        self,
        inputs: np.ndarray | Image.Image,
        aligned_size: Tuple[int, int],
        ratios: Tuple[float, float],
        *args,
        **kwargs,
    ) -> np.ndarray:
        if isinstance(inputs, Image.Image):
            inputs = np.array(inputs)
        aligned_height, aligned_width = aligned_size
        height_ratio, width_ratio = ratios

        interpolation = self.interpolation
        if self.pad_location == "edge" and height_ratio == width_ratio and height_ratio != 1:
            interpolation = INTER_LINEAR if height_ratio > 1 else INTER_AREA

        resized_image = cv2.resize(inputs, (aligned_width, aligned_height), interpolation=interpolation)
        return self._pad(np.array(resized_image), (aligned_height, aligned_width))

    def _pad(self, inputs: np.ndarray, aligned_size: Tuple[int, int]) -> np.ndarray:
        """padding image.

        Args:
            inputs (np.ndarray): image.
            aligned_size (Tuple[int, int]): aligend size.

        Raises:
            ValueError: if pad_location is invalid, raise ValueError.

        Returns:
            np.ndarray: padded resize.
        """
        height_pad = self.height - aligned_size[0]
        width_pad = self.width - aligned_size[1]

        if self.pad_location == "edge":
            height_pad /= 2
            width_pad /= 2

            top = int(round(height_pad - 0.1))
            bottom = int(round(height_pad + 0.1))
            left = int(round(width_pad - 0.1))
            right = int(round(width_pad + 0.1))
        else:
            raise ValueError(f"Invalid pad_location value. {self.pad_location}")
        image = cv2.copyMakeBorder(inputs, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.pad_value)
        return image


class Resize:
    """Resize the input to the given size.

    Args:
        mode (ResizeMode): Resize mode. Resize mode determines the resizing method.
        size (int | List[int]): target size.
        interpolation (Interpolation): interpolation method.
        pad_location
        pad_value
    """

    def __init__(self, mode: Optional[ResizeMode] = None, **args) -> None:
        if mode is None:
            mode = ResizeMode.default
        self._check_mode(mode)
        self.resize_args = ResizeArgs(**self._get_default_args_for_mode(mode, args))
        self._check_interpolation(self.resize_args.interpolation)

        self.resize_method = self._get_resize_method(mode)

    def _check_mode(self, mode: ResizeMode) -> None:
        """Check if the input mode is valid.
        The mode must be one of ['cv2', 'torchvision'].

        Args:
            mode (ResizeMode): input mode.

        Raises:
            ValueError: if input mode is invalid, raise ValueError.
        """
        if not ResizeMode.has_value(mode):
            raise ValueError(f"Invalid mode. {mode}")

    def _get_default_args_for_mode(self, mode: ResizeMode, args: dict):
        if args.get(ResizeArgEnum.backend) is None:
            args[ResizeArgEnum.backend] = ARGS_FOR_MODE[mode][ResizeArgEnum.backend]

        if args.get(ResizeArgEnum.align_side) is None:
            args[ResizeArgEnum.align_side] = ARGS_FOR_MODE[mode][ResizeArgEnum.align_side]

        if args.get(ResizeArgEnum.scale_method) is None:
            args[ResizeArgEnum.scale_method] = ARGS_FOR_MODE[mode][ResizeArgEnum.scale_method]

        if args.get(ResizeArgEnum.interpolation) is None:
            args[ResizeArgEnum.interpolation] = ARGS_FOR_MODE[mode][ResizeArgEnum.interpolation]
        return args

    def _check_interpolation(self, interpolation: InterpolationEnum) -> None:
        if not InterpolationEnum.has_value(interpolation):
            raise ValueError(f"Invalid Interpolation Mode. {interpolation}")

    def _get_resize_method(self, mode: ResizeMode) -> Callable:
        """get reize method based on input mode.
        if the input mode is "cv2", it return cv2.resize method.
        if the input mode is "torchvision" it return torchvision resize method.

        Args:
            mode (ResizeMode): input mode.

        Raises:
            ValueError: if the input mode is invalid, raise ValueErorr.

        Returns:
            Callable: resize mehtod.
        """
        if mode == ResizeMode.default or mode == ResizeMode.pycls:
            return CV2Resize(**self.resize_args.__dict__)
        elif mode == ResizeMode.torchvision:
            return TorchVisionResize(**self.resize_args.__dict__)
        elif mode == ResizeMode.pad:
            return PadResize(**self.resize_args.__dict__)
        else:
            raise ValueError(f"Invalid Resize mode. {mode}")

    def _get_raitos(self, image_size: List[int]) -> Tuple[float, float]:
        height, width = image_size
        short, long = min(height, width), max(height, width)
        if self.resize_args.align_side == AlignSideEnum.short:
            ratio_value = (short, short)
        elif self.resize_args.align_side == AlignSideEnum.long:
            ratio_value = (long, long)
        elif self.resize_args.align_side == AlignSideEnum.both:
            ratio_value = (height, width)
        else:
            raise ValueError(
                f"Should select item one of {AlignSideEnum._member_names_} not {self.resize_args.align_side}"
            )
        return ratio_value

    def _alingn_size(self, image_size: List[int]) -> Tuple[Tuple[int, int], Tuple[float, float]]:
        """aling image size.
        it alingn image size to origin image's long side.

        Args:
            image_size (List[int]): origin image size.

        Returns:
            Tuple[Tuple[int, int], Tuple[float, float]]: alinged size.
        """
        # image size: Height, Width
        height, width = image_size
        ratio_value = self._get_raitos(image_size)

        height_ratio = self.resize_args.size[0] / ratio_value[0]
        width_ratio = self.resize_args.size[1] / ratio_value[1]

        alingned_height = int(height_ratio * height + EPSILON)
        alingned_width = int(width_ratio * width + EPSILON)

        return (alingned_height, alingned_width), (height_ratio, width_ratio)

    def __call__(self, inputs: np.ndarray | Image.Image) -> np.ndarray:
        """Resize the input image to target size."""
        if isinstance(inputs, Image.Image):
            image_size = (inputs.size[1], inputs.size[0])
        else:
            image_size = inputs.shape[:2]
        alined_size, ratios = self._alingn_size(image_size)
        return self.resize_method(inputs, alined_size, ratios)
