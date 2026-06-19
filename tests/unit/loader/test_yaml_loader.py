"""Tests for dx_modelzoo.loader.yaml_loader."""

import pytest
import yaml

from dx_modelzoo.common.exceptions import YamlValidationError
from dx_modelzoo.loader.yaml_loader import load_yaml, validate_yaml


MINIMAL_VALID_CONFIG = {
    "name": "test_model",
    "inputs": [{"name": "input", "shape": [1, 3, 224, 224], "dtype": "float32"}],
    "preprocessing": [{"type": "div", "x": 255}],
    "postprocessing": [{"type": "topk", "k": [1, 5]}],
    "dataset": {"type": "ILSVRC2012", "eval_path": "/data"},
    "artifacts": {"path": "/models/test"},
    "profiles": {
        "onnx": {
            "target": "onnx",
            "runtime": {"device": "gpu"},
        }
    },
}


class TestLoadYaml:
    def test_loads_valid_yaml(self, tmp_path):
        p = tmp_path / "model.yaml"
        p.write_text(yaml.dump(MINIMAL_VALID_CONFIG))
        config = load_yaml(p)
        assert config["name"] == "test_model"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_yaml(tmp_path / "nonexistent.yaml")

    def test_non_dict_raises(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("- just a list")
        with pytest.raises(YamlValidationError, match="root must be a mapping"):
            load_yaml(p)


class TestValidateYaml:
    def test_valid_config_passes(self):
        validate_yaml(MINIMAL_VALID_CONFIG, "test.yaml")

    def test_missing_required_fields(self):
        config = {"name": "test"}
        with pytest.raises(YamlValidationError, match="Missing required fields"):
            validate_yaml(config, "test.yaml")

    def test_missing_input_shape(self):
        config = {**MINIMAL_VALID_CONFIG, "inputs": [{"name": "x"}]}
        with pytest.raises(YamlValidationError, match="missing 'shape'"):
            validate_yaml(config, "test.yaml")

    def test_unknown_preprocessing_type(self):
        config = {**MINIMAL_VALID_CONFIG, "preprocessing": [{"type": "nonexistent_op"}]}
        with pytest.raises(YamlValidationError, match="Unknown preprocessing type"):
            validate_yaml(config, "test.yaml")

    def test_empty_profiles_raises(self):
        config = {**MINIMAL_VALID_CONFIG, "profiles": {}}
        with pytest.raises(YamlValidationError, match="At least one profile"):
            validate_yaml(config, "test.yaml")

    def test_profile_missing_runtime(self):
        config = {**MINIMAL_VALID_CONFIG, "profiles": {"p1": {"target": "onnx"}}}
        with pytest.raises(YamlValidationError, match="missing 'runtime'"):
            validate_yaml(config, "test.yaml")

    def test_profile_missing_device(self):
        config = {**MINIMAL_VALID_CONFIG, "profiles": {"p1": {"target": "onnx", "runtime": {}}}}
        with pytest.raises(YamlValidationError, match="missing 'device'"):
            validate_yaml(config, "test.yaml")

    def test_quantization_invalid_keys(self):
        config = {
            **MINIMAL_VALID_CONFIG,
            "profiles": {
                "q": {
                    "target": "dxnn",
                    "runtime": {"device": 0},
                    "compile": {"quantization": {"invalid_key": {}}},
                }
            },
        }
        with pytest.raises(YamlValidationError, match="invalid keys"):
            validate_yaml(config, "test.yaml")

    def test_quantization_mix_modes_raises(self):
        config = {
            **MINIMAL_VALID_CONFIG,
            "profiles": {
                "q": {
                    "target": "dxnn",
                    "runtime": {"device": 0},
                    "compile": {"quantization": {"lite": {"method": "ema"}, "pro": {"method": "ema"}}},
                }
            },
        }
        with pytest.raises(YamlValidationError, match="cannot mix"):
            validate_yaml(config, "test.yaml")

    def test_ppu_config_type0_valid(self):
        config = {
            **MINIMAL_VALID_CONFIG,
            "profiles": {
                "q": {
                    "target": "dxnn",
                    "runtime": {"device": 0},
                    "compile": {
                        "quantization": {"lite": {"method": "ema", "num_samples": 100}},
                        "ppu_config": {
                            "type": 0,
                            "num_classes": 80,
                            "conf_thres": 0.5,
                            "activation": "sigmoid",
                            "layer": {"head0": {"num_anchors": 3}},
                        },
                    },
                }
            },
        }
        validate_yaml(config, "test.yaml")  # Should not raise

    def test_ppu_config_type1_valid(self):
        config = {
            **MINIMAL_VALID_CONFIG,
            "profiles": {
                "q": {
                    "target": "dxnn",
                    "runtime": {"device": 0},
                    "compile": {
                        "quantization": {"lite": {}},
                        "ppu_config": {
                            "type": 1,
                            "num_classes": 80,
                            "conf_thres": 0.5,
                            "layer": [{"bbox": "output0", "cls_conf": "output1"}],
                        },
                    },
                }
            },
        }
        validate_yaml(config, "test.yaml")

    def test_ppu_config_type2_valid(self):
        config = {
            **MINIMAL_VALID_CONFIG,
            "profiles": {
                "q": {
                    "target": "dxnn",
                    "runtime": {"device": 0},
                    "compile": {
                        "quantization": {"lite": {}},
                        "ppu_config": {
                            "type": 2,
                            "num_classes": 80,
                            "topk": 100,
                            "layer": [{"bbox": "out0", "cls_conf": "out1"}],
                        },
                    },
                }
            },
        }
        validate_yaml(config, "test.yaml")

    def test_ppu_config_invalid_type(self):
        config = {
            **MINIMAL_VALID_CONFIG,
            "profiles": {
                "q": {
                    "target": "dxnn",
                    "runtime": {"device": 0},
                    "compile": {
                        "quantization": {"lite": {}},
                        "ppu_config": {"type": 5, "num_classes": 80, "layer": []},
                    },
                }
            },
        }
        with pytest.raises(YamlValidationError, match="must be 0, 1, or 2"):
            validate_yaml(config, "test.yaml")

    def test_pipeline_yaml_valid(self):
        config = {
            "name": "pipeline_model",
            "pipeline": {
                "det": {
                    "inputs": [{"name": "img", "shape": [1, 3, 640, 640]}],
                    "preprocessing": [{"type": "resize", "size": [640, 640]}],
                    "artifacts": {"path": "/models/det"},
                },
                "rec": {
                    "inputs": [{"name": "crop", "shape": [1, 3, 32, 100]}],
                    "preprocessing": [{"type": "resize", "size": [32, 100]}],
                    "artifacts": {"path": "/models/rec"},
                },
            },
            "dataset": {"type": "COCO"},
            "profiles": {"onnx": {"target": "onnx", "runtime": {"device": "gpu"}}},
        }
        validate_yaml(config, "test.yaml")

    def test_pipeline_yaml_missing_stage_fields(self):
        config = {
            "name": "bad_pipeline",
            "pipeline": {"det": {"inputs": [{"name": "x", "shape": [1, 3, 640, 640]}]}},
            "dataset": {"type": "COCO"},
            "profiles": {"onnx": {"target": "onnx", "runtime": {"device": "gpu"}}},
        }
        with pytest.raises(YamlValidationError, match="Missing required fields"):
            validate_yaml(config, "test.yaml")

    def test_runtime_async_bool_validation(self):
        config = {
            **MINIMAL_VALID_CONFIG,
            "profiles": {"p": {"target": "dxnn", "runtime": {"device": 0, "async": "yes"}}},
        }
        with pytest.raises(YamlValidationError, match="async must be a boolean"):
            validate_yaml(config, "test.yaml")

    def test_runtime_buffer_count_validation(self):
        config = {
            **MINIMAL_VALID_CONFIG,
            "profiles": {"p": {"target": "dxnn", "runtime": {"device": 0, "buffer_count": 200}}},
        }
        with pytest.raises(YamlValidationError, match="buffer_count"):
            validate_yaml(config, "test.yaml")

    def test_quantization_p_level_valid(self):
        config = {
            **MINIMAL_VALID_CONFIG,
            "profiles": {
                "q": {
                    "target": "dxnn",
                    "runtime": {"device": 0},
                    "compile": {"quantization": {"p2": None, "p5": None}},
                }
            },
        }
        validate_yaml(config, "test.yaml")

    def test_quantization_invalid_p_combo(self):
        config = {
            **MINIMAL_VALID_CONFIG,
            "profiles": {
                "q": {
                    "target": "dxnn",
                    "runtime": {"device": 0},
                    "compile": {"quantization": {"p0": None, "p5": None}},
                }
            },
        }
        with pytest.raises(YamlValidationError, match="invalid.*p-level"):
            validate_yaml(config, "test.yaml")
