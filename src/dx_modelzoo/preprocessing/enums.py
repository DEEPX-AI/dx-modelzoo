from enum import IntEnum

try:
    from enum import StrEnum
except ImportError:
    from enum import Enum

    class StrEnum(str, Enum):
        pass


class ResizeMode(StrEnum):
    torchvision = "torchvision"
    default = "default"
    pad = "pad"
    pycls = "pycls"

    @classmethod
    def has_value(cls, value: str) -> bool:
        return value in cls._value2member_map_


class ResizeArgEnum(StrEnum):
    size = "size"
    interpolation = "interpolation"
    backend = "backend"
    align_side = "align_side"
    scale_method = "scale_method"
    pad_location = "pad_location"
    pad_value = "pad_value"


class BackendEnum(StrEnum):
    cv2 = "cv2"
    pil = "pil"


class AlignSideEnum(StrEnum):
    both = "both"
    long = "long"
    short = "short"


class ScaleMethodEnum(StrEnum):
    scale_up = "scale_up"
    scale_down = "scale_down"


class InterpolationEnum(StrEnum):
    BILINEAR = "BILINEAR"
    LINEAR = "LINEAR"
    NEAREST = "NEAREST"
    BICUBIC = "BICUBIC"

    @classmethod
    def has_value(cls, value: str) -> bool:
        return value in cls._value2member_map_


class PILResizeInterpolationEnum(IntEnum):
    NEAREST = 0
    LANCZOS = 1
    BILINEAR = 2
    LINEAR = 2
    BICUBIC = 3
    BOX = 4
    HAMMING = 5


class CVResizeInterpolationEnum(IntEnum):
    NEAREST = 0
    LINEAR = 1
    BILINEAR = 1
    BICUBIC = 2
    AREA = 3
    LANCZOS4 = 4
