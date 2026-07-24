"""Tests for dx_modelzoo.session.factory."""

import pytest

from dx_modelzoo.session.factory import SessionCreationError, _detect_target


class TestDetectTarget:
    def test_onnx_extension(self):
        assert _detect_target("model.onnx") == "onnx"

    def test_dxnn_extension(self):
        assert _detect_target("model.dxnn") == "dxnn"

    def test_unknown_extension_raises(self):
        with pytest.raises(SessionCreationError, match="Cannot determine"):
            _detect_target("model.pt")


class TestCreateSession:
    def test_missing_onnx_file_raises(self, tmp_path):
        from dx_modelzoo.session.factory import create_session
        with pytest.raises(SessionCreationError, match="not found"):
            create_session(str(tmp_path / "missing.onnx"))

    def test_profile_without_builder_raises(self):
        from dx_modelzoo.session.factory import create_session
        with pytest.raises(SessionCreationError, match="not a file path"):
            create_session("q-lite")
