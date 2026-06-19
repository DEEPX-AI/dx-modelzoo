"""Tests for dx_modelzoo.common.enums."""

from dx_modelzoo.common.enums import DatasetType, DeviceType, EvaluationType, SessionType


class TestDeviceType:
    def test_values(self):
        assert DeviceType.GPU == "gpu"
        assert DeviceType.CPU == "cpu"
        assert DeviceType.NPU == "npu"

    def test_is_npu(self):
        assert DeviceType.NPU.is_npu() is True
        assert DeviceType.GPU.is_npu() is False
        assert DeviceType.CPU.is_npu() is False

    def test_str_comparison(self):
        assert DeviceType("gpu") == DeviceType.GPU


class TestSessionType:
    def test_values(self):
        assert SessionType.onnxruntime == "OnnxRuntime"
        assert SessionType.simulator == "Simulator"
        assert SessionType.dxruntime == "DxRuntime"


class TestEvaluationType:
    def test_metric_returns_known(self):
        assert "TopK1" in EvaluationType.image_classification.metric()
        assert "mAP" in EvaluationType.coco.metric()
        assert "mIoU" in EvaluationType.segmentation.metric()
        assert "PSNR" in EvaluationType.bsd68.metric()
        assert "RMSE" in EvaluationType.depth_estimation.metric()

    def test_metric_all_defined(self):
        for member in EvaluationType:
            result = member.metric()
            assert result != "Unknown", f"{member.name} has no metric mapping"


class TestDatasetType:
    def test_all_values_are_strings(self):
        for member in DatasetType:
            assert isinstance(member.value, str)

    def test_known_members(self):
        assert DatasetType.COCO == "COCO"
        assert DatasetType.ILSVRC2012 == "ILSVRC2012"
        assert DatasetType.Cityscapes == "Cityscapes"
