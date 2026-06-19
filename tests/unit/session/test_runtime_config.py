"""Tests for dx_modelzoo.session.runtime_config."""

from dx_modelzoo.session.runtime_config import (
    DxnnRuntimeConfig,
    OnnxRuntimeConfig,
    RuntimeConfig,
)


class TestRuntimeConfig:
    def test_defaults(self):
        cfg = RuntimeConfig()
        assert cfg.device is None
        assert cfg.batch_size == 1
        assert cfg.async_mode is None
        assert cfg.use_async is False  # ASYNC_DEFAULT = False

    def test_from_profile_onnx(self):
        profile = {"target": "onnx", "runtime": {"device": "gpu", "batch_size": 4}}
        cfg = RuntimeConfig.from_profile(profile)
        assert isinstance(cfg, OnnxRuntimeConfig)
        assert cfg.device == "gpu"
        assert cfg.batch_size == 4

    def test_from_profile_dxnn(self):
        profile = {"target": "dxnn", "runtime": {"device": 0, "buffer_count": 8, "use_ort": False}}
        cfg = RuntimeConfig.from_profile(profile)
        assert isinstance(cfg, DxnnRuntimeConfig)
        assert cfg.device == 0
        assert cfg.buffer_count == 8
        assert cfg.use_ort is False

    def test_from_profile_none(self):
        cfg = RuntimeConfig.from_profile(None)
        assert isinstance(cfg, OnnxRuntimeConfig)

    def test_with_device_override(self):
        cfg = RuntimeConfig(device="cpu")
        new_cfg = cfg.with_device_override("gpu")
        assert new_cfg.device == "gpu"
        assert cfg.device == "cpu"  # original unchanged

    def test_with_device_override_none(self):
        cfg = RuntimeConfig(device="cpu")
        same = cfg.with_device_override(None)
        assert same is cfg


class TestDxnnRuntimeConfig:
    def test_async_default_true(self):
        cfg = DxnnRuntimeConfig()
        assert cfg.use_async is True  # DXNN defaults to async

    def test_explicit_async_false(self):
        cfg = DxnnRuntimeConfig(async_mode=False)
        assert cfg.use_async is False

    def test_from_runtime(self):
        runtime = {"device": 0, "batch_size": 2, "buffer_count": 4, "use_ort": True}
        cfg = DxnnRuntimeConfig.from_runtime(runtime)
        assert cfg.buffer_count == 4
        assert cfg.use_ort is True
        assert cfg.batch_size == 2


class TestOnnxRuntimeConfig:
    def test_async_default_false(self):
        cfg = OnnxRuntimeConfig()
        assert cfg.use_async is False
