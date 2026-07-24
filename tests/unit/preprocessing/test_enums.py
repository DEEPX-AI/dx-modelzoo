"""Tests for dx_modelzoo.preprocessing.enums."""

from dx_modelzoo.preprocessing.enums import AlignSideEnum, BackendEnum, ResizeMode


class TestResizeMode:
    def test_values(self):
        assert ResizeMode.torchvision == "torchvision"
        assert ResizeMode.default == "default"
        assert ResizeMode.pad == "pad"
        assert ResizeMode.pycls == "pycls"

    def test_has_value(self):
        assert ResizeMode.has_value("torchvision") is True
        assert ResizeMode.has_value("nonexistent") is False


class TestBackendEnum:
    def test_values(self):
        assert BackendEnum.cv2 == "cv2"
        assert BackendEnum.pil == "pil"


class TestAlignSideEnum:
    def test_values(self):
        assert AlignSideEnum.both == "both"
        assert AlignSideEnum.long == "long"
