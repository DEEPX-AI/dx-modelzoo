"""Tests for dx_modelzoo.command.compile."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from dx_modelzoo.command.compile import _make_calibration_loader, _remove_expanddim, compile_model


class TestRemoveExpanddim:
    def test_removes_expanddim_from_list(self):
        config = [
            {"type": "resize", "size": [224, 224]},
            {"type": "expanddim", "axis": 0},
            {"type": "div", "x": 255},
        ]
        result = _remove_expanddim(config)
        types = [s["type"] for s in result]
        assert "expanddim" not in types
        assert "resize" in types
        assert "div" in types

    def test_no_expanddim(self):
        config = [{"type": "resize", "size": [224, 224]}]
        result = _remove_expanddim(config)
        assert len(result) == 1

    def test_dict_config(self):
        config = {
            "input0": [{"type": "expanddim", "axis": 0}, {"type": "div", "x": 255}],
            "input1": [{"type": "resize", "size": [32, 32]}],
        }
        result = _remove_expanddim(config)
        assert isinstance(result, dict)
        assert all(s["type"] != "expanddim" for s in result["input0"])
        assert len(result["input1"]) == 1

    def test_non_list_non_dict(self):
        assert _remove_expanddim(None) is None
        assert _remove_expanddim("string") == "string"


class CalibrationDataset:
    def __init__(self, item):
        self.item = item

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.item


def test_calibration_preserves_integer_dtype_and_precision():
    import torch

    value = np.array([2**53 + 1], dtype=np.int64)
    dataset = CalibrationDataset(({"input": value}, 0))
    loader = _make_calibration_loader(
        dataset,
        num_samples=1,
        input_specs=[{"name": "input", "shape": [1, 1], "dtype": "int64"}],
    )
    batch = next(iter(loader))
    assert batch["input"].dtype == torch.int64
    assert int(batch["input"].item()) == 2**53 + 1


def test_calibration_rejects_dataset_constant_input_collision():
    dataset = CalibrationDataset(({"input": np.array([1]), "threshold": np.array([9])}, 0))
    loader = _make_calibration_loader(
        dataset,
        num_samples=1,
        input_specs=[
            {"name": "input", "shape": [1, 1], "dtype": "int32"},
            {"name": "threshold", "shape": [1, 1], "dtype": "int32", "value": 7},
        ],
    )
    with pytest.raises(ValueError, match="constant input keys"):
        next(iter(loader))


def test_compile_passes_profile_attributes_unchanged(monkeypatch, tmp_path):
    captured = {}

    def compile_fn(**kwargs):
        captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "dx_com", SimpleNamespace(compile=compile_fn))

    class Builder:
        name = "test_model"
        yaml_path = tmp_path / "test_model.yaml"
        model_dir = tmp_path
        is_pipeline = False
        config = {
            "artifacts": {"path": str(tmp_path)},
            "preprocessing": [],
            "inputs": [{"name": "input", "shape": [1, 3, 2, 2], "dtype": "float32"}],
        }

        def get_profile(self, profile_name):
            assert profile_name == "dxnn-profile"
            return {
                "compile": {
                    "use_gpu": True,
                    "verbose": False,
                    "model": "yaml-model-must-not-win",
                    "output_dir": "yaml-output-must-not-win",
                    "config": "yaml-config-must-not-win",
                    "dataloader": "yaml-loader-must-not-win",
                    "calibration": {"num": 7, "method": "ema"},
                    "pre_optimize_config": [{"custom_pass": {"enabled": True}}],
                }
            }

        def build_dataset(self):
            class Dataset:
                preprocessing = None

                def __len__(self):
                    return 0

            return Dataset()

    onnx_path = tmp_path / "test_model.onnx"
    onnx_path.touch()
    compile_model(Builder(), "dxnn-profile", model_path=str(onnx_path))

    assert captured["model"] == str(onnx_path)
    assert captured["output_dir"] == str(tmp_path)
    assert captured["config"] is None
    assert captured["dataloader"] is not None
    assert captured["use_gpu"] is True
    assert captured["calibration"] == {"num": 7, "method": "ema"}
    assert captured["pre_optimize_config"] == [{"custom_pass": {"enabled": True}}]


def test_compile_defaults_export_html_to_true(monkeypatch, tmp_path):
    captured = {}

    def compile_fn(**kwargs):
        captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "dx_com", SimpleNamespace(compile=compile_fn))

    class Builder:
        name = "test_model"
        yaml_path = tmp_path / "test_model.yaml"
        model_dir = tmp_path
        is_pipeline = False
        config = {
            "artifacts": {"path": str(tmp_path)},
            "preprocessing": [],
            "inputs": [{"name": "input", "shape": [1, 3, 2, 2], "dtype": "float32"}],
        }

        def get_profile(self, profile_name):
            return {"compile": {}}

        def build_dataset(self):
            class Dataset:
                preprocessing = None

                def __len__(self):
                    return 0

            return Dataset()

    onnx_path = tmp_path / "test_model.onnx"
    onnx_path.touch()
    compile_model(Builder(), "q-lite", model_path=str(onnx_path))

    assert captured["export_html"] is True
