"""Tests for dx_modelzoo.main (CLI entry point)."""

import os

import pytest
import yaml
from typer.testing import CliRunner

from dx_modelzoo.main import app, _apply_env_overrides, _parse_model_path

runner = CliRunner()


class TestApplyEnvOverrides:
    def test_sets_data_root(self, monkeypatch):
        monkeypatch.delenv("DATA_ROOT", raising=False)
        _apply_env_overrides(data_root="/my/data")
        assert os.environ["DATA_ROOT"] == "/my/data"

    def test_sets_model_root(self, monkeypatch):
        monkeypatch.delenv("MODEL_ROOT", raising=False)
        _apply_env_overrides(model_root="/my/models")
        assert os.environ["MODEL_ROOT"] == "/my/models"

    def test_none_does_not_override(self, monkeypatch):
        monkeypatch.setenv("DATA_ROOT", "original")
        _apply_env_overrides(data_root=None)
        assert os.environ["DATA_ROOT"] == "original"


class TestParseModelPath:
    def test_none_returns_none(self):
        assert _parse_model_path(None) is None

    def test_plain_path(self):
        assert _parse_model_path("/path/to/model.onnx") == "/path/to/model.onnx"

    def test_stage_path_format(self):
        result = _parse_model_path("det=/path/det.onnx,rec=/path/rec.onnx")
        assert isinstance(result, dict)
        assert result["det"] == "/path/det.onnx"
        assert result["rec"] == "/path/rec.onnx"

    def test_invalid_format_raises(self):
        import typer
        with pytest.raises(typer.BadParameter):
            _parse_model_path("det=/a,bad_format")


class TestCLIList:
    def test_list_command_runs(self):
        result = runner.invoke(app, ["list", "--all"])
        # Should not crash (exit 0 or display models)
        assert result.exit_code == 0

    def test_list_with_task_filter(self):
        result = runner.invoke(app, ["list", "--all", "--task", "image_classification"])
        assert result.exit_code == 0


class TestCLIInfo:
    def test_info_with_yaml(self, tmp_path):
        yaml_cfg = {
            "name": "testmodel",
            "inputs": [{"name": "input", "shape": [1, 3, 224, 224], "dtype": "float32"}],
            "preprocessing": [{"type": "div", "x": 255}],
            "dataset": {"type": "ILSVRC2012", "eval_path": "/data"},
            "artifacts": {"path": "/models"},
            "profiles": {"onnx": {"target": "onnx", "runtime": {"device": "gpu"}}},
        }
        yaml_path = tmp_path / "testmodel.yaml"
        yaml_path.write_text(yaml.dump(yaml_cfg))
        result = runner.invoke(app, ["info", str(yaml_path)])
        assert result.exit_code == 0
        assert "testmodel" in result.output


class TestAppExists:
    def test_app_is_typer(self):
        import typer
        assert isinstance(app, typer.Typer)

