from enum import IntEnum, StrEnum

import cv2
import numpy as np
from PIL import Image


class ColorConvertForm(StrEnum):
    BGR2RGB = "BGR2RGB"
    BGR2GRAY = "BGR2GRAY"

    @classmethod
    def has_value(cls, value: str) -> bool:
        return value in cls._value2member_map_


class Cv2CorlorConversionCode(IntEnum):
    BGR2BGRA = 0
    BGRA2BGR = 1
    BGR2RGBA = 3
    BGR2RGB = 4
    BGR2GRAY = 6


class PILColorConversionMode(StrEnum):
    BGR2RGB = "RGB"


class ConvertColor:
    """Converts the image from one color space to another.

    Args:
        form (ColorConvertForm): color space conversion form.
    """

    def __init__(self, form: ColorConvertForm) -> None:
        self._check_form(form)
        self.form = form

    def _check_form(self, form: ColorConvertForm) -> None:
        """check if the input form is valid.
        the input form Must be one of ["BGR2RGB"]

        Args:
            form (ColorConvertForm): input form

        Raises:
            ValueError: if input form is invalid, rais ValueError.
        """
        if not ColorConvertForm.has_value(form):
            raise ValueError(f"Invalid form. {form}")

    def __call__(self, inputs: np.ndarray | Image.Image) -> np.ndarray:
        """Convert input image's color space.

        Args:
            inputs (np.ndarray): input image.

        Returns:
            np.ndarray: converted image.
        """
        if isinstance(inputs, np.ndarray):
            if inputs.dtype == np.uint8:
                return cv2.cvtColor(inputs, code=Cv2CorlorConversionCode[self.form])
            else:
                return np.ascontiguousarray(inputs[:, :, ::-1])
        elif isinstance(inputs, Image.Image):
            return np.array(inputs.convert(PILColorConversionMode[self.form]))
