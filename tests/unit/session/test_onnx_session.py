"""Tests for dx_modelzoo.session.onnx_session."""

import numpy as np
import pytest


class TestOnnxRuntimeSession:
    def test_init_with_nonexistent_path_raises(self):
        from dx_modelzoo.session.onnx_session import OnnxRuntimeSession
        with pytest.raises(Exception):
            OnnxRuntimeSession("/nonexistent/model.onnx")

    def test_get_ort_provider_cpu(self):
        from dx_modelzoo.session.onnx_session import _get_ort_provider
        providers = _get_ort_provider("cpu")
        assert isinstance(providers, list)
        assert "CPUExecutionProvider" in providers

    def test_get_ort_provider_gpu(self):
        from dx_modelzoo.session.onnx_session import _get_ort_provider
        providers = _get_ort_provider("gpu")
        assert isinstance(providers, list)
        assert len(providers) > 0

    def test_get_ort_provider_none(self):
        from dx_modelzoo.session.onnx_session import _get_ort_provider
        providers = _get_ort_provider(None)
        assert "CPUExecutionProvider" in providers

    def test_get_ort_provider_int_device(self):
        from dx_modelzoo.session.onnx_session import _get_ort_provider
        providers = _get_ort_provider(0)
        assert isinstance(providers, list)

    def test_get_ort_provider_list_device(self):
        from dx_modelzoo.session.onnx_session import _get_ort_provider
        providers = _get_ort_provider([0])
        assert isinstance(providers, list)

    def test_get_ort_session_options(self):
        from dx_modelzoo.session.onnx_session import _get_ort_session_options
        opts = _get_ort_session_options()
        assert opts is not None

    def test_preload_nvidia_libs_no_crash(self):
        from dx_modelzoo.session.onnx_session import _preload_nvidia_libs
        # Should not raise even if nvidia packages not installed
        _preload_nvidia_libs()

