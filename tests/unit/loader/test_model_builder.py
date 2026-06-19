"""Tests for dx_modelzoo.loader.model_builder."""

import yaml
import pytest

from dx_modelzoo.loader.model_builder import ModelBuilder


MINIMAL_YAML = {
    "name": "test_model_224x224",
    "task": "image_classification",
    "inputs": [{"name": "input", "shape": [1, 3, 224, 224], "dtype": "float32", "layout": "NCHW"}],
    "preprocessing": [
        {"type": "resize", "size": [224, 224]},
        {"type": "div", "x": 255},
        {"type": "normalize", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        {"type": "transpose", "axis": [2, 0, 1]},
        {"type": "expanddim", "axis": 0},
    ],
    "postprocessing": [{"type": "topk", "k": [1, 5]}],
    "dataset": {"type": "ILSVRC2012", "eval_path": "${DATA_ROOT}/ILSVRC2012/val"},
    "artifacts": {"path": "${MODEL_ROOT}/${MODEL_NAME}"},
    "profiles": {
        "onnx": {"target": "onnx", "runtime": {"device": "gpu", "batch_size": 1}},
        "q-lite": {
            "target": "dxnn",
            "compile": {"quantization": {"lite": {"method": "ema", "num_samples": 100}}},
            "runtime": {"device": 0},
        },
    },
}


class TestModelBuilder:
    def _write_yaml(self, tmp_path, config=None):
        model_dir = tmp_path / "cv" / "image_classification" / "test"
        model_dir.mkdir(parents=True)
        yaml_path = model_dir / "test_model_224x224.yaml"
        yaml_path.write_text(yaml.dump(config or MINIMAL_YAML))
        return yaml_path

    def test_init_loads_config(self, tmp_path):
        yaml_path = self._write_yaml(tmp_path)
        builder = ModelBuilder(yaml_path, resolve_env=False)
        assert builder.name == "test_model_224x224"

    def test_inputs_property(self, tmp_path):
        yaml_path = self._write_yaml(tmp_path)
        builder = ModelBuilder(yaml_path, resolve_env=False)
        assert len(builder.inputs) == 1
        assert builder.inputs[0]["shape"] == [1, 3, 224, 224]

    def test_get_profile(self, tmp_path):
        yaml_path = self._write_yaml(tmp_path)
        builder = ModelBuilder(yaml_path, resolve_env=False)
        profile = builder.get_profile("onnx")
        assert profile["target"] == "onnx"

    def test_get_profile_missing_raises(self, tmp_path):
        yaml_path = self._write_yaml(tmp_path)
        builder = ModelBuilder(yaml_path, resolve_env=False)
        with pytest.raises(KeyError, match="not found"):
            builder.get_profile("nonexistent")

    def test_is_pipeline_false(self, tmp_path):
        yaml_path = self._write_yaml(tmp_path)
        builder = ModelBuilder(yaml_path, resolve_env=False)
        assert builder.is_pipeline is False
        assert builder.pipeline_stages == []

    def test_appendix_empty(self, tmp_path):
        yaml_path = self._write_yaml(tmp_path)
        builder = ModelBuilder(yaml_path, resolve_env=False)
        assert builder.appendix == {}

    def test_build_preprocessing(self, tmp_path):
        yaml_path = self._write_yaml(tmp_path)
        builder = ModelBuilder(yaml_path, resolve_env=False)
        prep = builder.build_preprocessing("onnx")
        import numpy as np
        img = (np.random.rand(300, 300, 3) * 255).astype(np.uint8)
        result = prep(img)
        assert result.shape == (1, 3, 224, 224)

    def test_build_postprocessing(self, tmp_path):
        yaml_path = self._write_yaml(tmp_path)
        builder = ModelBuilder(yaml_path, resolve_env=False)
        post = builder.build_postprocessing()
        import numpy as np
        logits = [np.random.rand(1, 1000).astype(np.float32)]
        result = post(logits)
        assert result.shape[-1] == 5  # top-5

    def test_build_dataset(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DATA_ROOT", str(tmp_path))
        yaml_path = self._write_yaml(tmp_path)
        builder = ModelBuilder(yaml_path, resolve_env=False)
        # build_dataset with explicit path (won't check if dir exists at this level)
        ds = builder.build_dataset(data_dir=str(tmp_path))
        from dx_modelzoo.common.dataloader import DatasetBase
        assert isinstance(ds, DatasetBase)

    def test_build_evaluator(self, tmp_path):
        yaml_path = self._write_yaml(tmp_path)
        builder = ModelBuilder(yaml_path, resolve_env=False)
        from dx_modelzoo.session import SessionBase
        class FakeSess(SessionBase):
            def __init__(self): super().__init__("/fake.onnx")
            def run(self, inputs): return [inputs]
        from dx_modelzoo.common.dataloader import DatasetBase
        class FakeDs(DatasetBase):
            def __init__(self): super().__init__("/f")
            def __len__(self): return 1
            def __getitem__(self, i): return None, None
        ev = builder.build_evaluator(FakeSess(), FakeDs(), "onnx")
        from dx_modelzoo.evaluator import EvaluatorBase
        assert isinstance(ev, EvaluatorBase)


PIPELINE_YAML = {
    "name": "ocr_pipeline",
    "pipeline": {
        "det": {
            "inputs": [{"name": "img", "shape": [1, 3, 640, 640], "dtype": "float32"}],
            "preprocessing": [{"type": "resize", "size": [640, 640]}, {"type": "div", "x": 255}],
            "artifacts": {"path": "${MODEL_ROOT}/det"},
        },
        "rec": {
            "inputs": [{"name": "crop", "shape": [1, 3, 32, 100], "dtype": "float32"}],
            "preprocessing": [{"type": "resize", "size": [32, 100]}, {"type": "div", "x": 255}],
            "artifacts": {"path": "${MODEL_ROOT}/rec"},
        },
    },
    "dataset": {"type": "COCO", "eval_path": "${DATA_ROOT}/COCO"},
    "profiles": {
        "onnx": {"target": "onnx", "runtime": {"device": "gpu"}},
    },
}


class TestModelBuilderPipeline:
    def _write_yaml(self, tmp_path, config=None):
        model_dir = tmp_path / "cv" / "ocr" / "pipeline"
        model_dir.mkdir(parents=True)
        yaml_path = model_dir / "ocr_pipeline.yaml"
        yaml_path.write_text(yaml.dump(config or PIPELINE_YAML))
        return yaml_path

    def test_is_pipeline(self, tmp_path):
        yaml_path = self._write_yaml(tmp_path)
        builder = ModelBuilder(yaml_path, resolve_env=False)
        assert builder.is_pipeline is True

    def test_pipeline_stages(self, tmp_path):
        yaml_path = self._write_yaml(tmp_path)
        builder = ModelBuilder(yaml_path, resolve_env=False)
        assert builder.pipeline_stages == ["det", "rec"]

    def test_get_stage_config(self, tmp_path):
        yaml_path = self._write_yaml(tmp_path)
        builder = ModelBuilder(yaml_path, resolve_env=False)
        det_cfg = builder.get_stage_config("det")
        assert det_cfg["inputs"][0]["shape"] == [1, 3, 640, 640]

    def test_inputs_from_first_stage(self, tmp_path):
        yaml_path = self._write_yaml(tmp_path)
        builder = ModelBuilder(yaml_path, resolve_env=False)
        inputs = builder.inputs
        assert inputs[0]["shape"] == [1, 3, 640, 640]

    def test_build_preprocessing_for_stage(self, tmp_path):
        yaml_path = self._write_yaml(tmp_path)
        builder = ModelBuilder(yaml_path, resolve_env=False)
        prep = builder.build_preprocessing("onnx", stage="det")
        import numpy as np
        img = (np.random.rand(800, 800, 3) * 255).astype(np.uint8)
        result = prep(img)
        assert result.shape[-2:] == (640, 640) or result.shape[0] == 640
