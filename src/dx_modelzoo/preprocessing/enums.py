from enum import IntEnum, StrEnum, auto


class ResizeMode(StrEnum):
    torchvision = auto()
    default = auto()
    pad = auto()
    pycls = auto()
    
    @classmethod
    def has_value(cls, value: str) -> bool:
        return value in cls._value2member_map_


class ResizeArgEnum(StrEnum):
    size = auto()
    interpolation = auto()
    backend = auto()
    align_side = auto()
    scale_method = auto()
    pad_location = auto()
    pad_value = auto()


class BackendEnum(StrEnum):
    cv2 = auto()
    pil = auto()


class AlignSideEnum(StrEnum):
    both = auto()
    long = auto()
    short = auto()


class ScaleMethodEnum(StrEnum):
    scale_up = auto()
    scale_down = auto()


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

    def __repr__(self) -> str:
        return self.name


class CVResizeInterpolationEnum(IntEnum):
    NEAREST = 0
    LINEAR = 1
    BILINEAR = 1
    CUBIC = 2
    AREA = 3
    LANCZOS4 = 4
