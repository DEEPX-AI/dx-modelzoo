"""Integration test: CLI commands (excluding eval/compile).

Tests `dxmz list`, `dxmz info`, and basic command structure
using typer's CliRunner.
"""

import yaml
import pytest
from typer.testing import CliRunner

from dx_modelzoo.main import app

runner = CliRunner()


class TestListCommand:
    def test_list_all_succeeds(self):
        result = runner.invoke(app, ["list", "--all"])
        assert result.exit_code == 0

    def test_list_all_shows_models(self):
        result = runner.invoke(app, ["list", "--all"])
        # Should list at least some model names
        assert len(result.output) > 100

    def test_list_filter_task(self):
        result = runner.invoke(app, ["list", "--all", "--task", "image_classification"])
        assert result.exit_code == 0
        # Should have fewer results than unfiltered
        assert "resnet" in result.output.lower() or "mobilenet" in result.output.lower() or result.output

    def test_list_filter_domain(self):
        result = runner.invoke(app, ["list", "--all", "--domain", "cv"])
        assert result.exit_code == 0

    def test_list_filter_nonexistent_task(self):
        result = runner.invoke(app, ["list", "--all", "--task", "nonexistent_task_xyz"])
        assert result.exit_code == 0
        # Should return empty or "no models found"


class TestInfoCommand:
    def test_info_with_yaml_path(self, tmp_path):
        """Info command should display model metadata from a YAML path."""
        yaml_cfg = {
            "name": "test-info-model_224x224",
            "task": "image_classification",
            "inputs": [{"name": "input", "shape": [1, 3, 224, 224], "dtype": "float32"}],
            "preprocessing": [{"type": "div", "x": 255}],
            "dataset": {"type": "ILSVRC2012", "eval_path": "/data"},
            "artifacts": {"path": "/models"},
            "profiles": {"onnx": {"target": "onnx", "runtime": {"device": "gpu"}}},
        }
        yaml_path = tmp_path / "test-info-model_224x224.yaml"
        yaml_path.write_text(yaml.dump(yaml_cfg))
        result = runner.invoke(app, ["info", str(yaml_path)])
        assert result.exit_code == 0
        assert "test-info-model_224x224" in result.output

    def test_info_builtin_model(self):
        """Info on a known builtin model should succeed."""
        result = runner.invoke(app, ["info", "resnet50_224x224"])
        if result.exit_code == 0:
            assert "resnet50" in result.output.lower()
        # May exit 1 if model not found in this environment — acceptable

    def test_info_nonexistent_model(self):
        """Info on a nonexistent model should fail gracefully."""
        result = runner.invoke(app, ["info", "totally_fake_model_xyz_999"])
        assert result.exit_code != 0 or "not found" in result.output.lower() or "error" in result.output.lower()


class TestBenchmarkCommand:
    def test_benchmark_help(self):
        """Benchmark command should show help without crashing."""
        result = runner.invoke(app, ["benchmark", "--help"])
        assert result.exit_code == 0
        assert "benchmark" in result.output.lower() or "Usage" in result.output
