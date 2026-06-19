"""Tests for dx_modelzoo.preprocessing (PreprocessingPipeline)."""

import numpy as np

from dx_modelzoo.preprocessing import PREPROCESSING_REGISTRY, PreprocessingPipeline


class TestPreprocessingPipeline:
    def test_basic_pipeline(self):
        config = [
            {"type": "div", "x": 255},
            {"type": "transpose", "axis": [2, 0, 1]},
        ]
        pipeline = PreprocessingPipeline(config)
        img = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
        result = pipeline(img)
        assert result.shape == (3, 32, 32)
        assert result.dtype == np.float32

    def test_npu_mode_skips_arithmetic(self):
        config = [
            {"type": "resize", "size": [64, 64]},
            {"type": "div", "x": 255},
            {"type": "normalize", "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
            {"type": "transpose", "axis": [2, 0, 1]},
            {"type": "expanddim", "axis": 0},
        ]
        pipeline = PreprocessingPipeline(config, npu_mode=True)
        # Only resize should remain (div, normalize, transpose, expanddim skipped)
        assert len(pipeline.steps) == 1  # resize only

    def test_dict_config_multimodal(self):
        config = {
            "input0": [{"type": "div", "x": 255}],
            "input1": [{"type": "normalize", "mean": [0.5], "std": [0.5]}],
        }
        pipeline = PreprocessingPipeline(config)
        # Steps from both inputs get flattened
        assert len(pipeline.steps) >= 2

    def test_steps_config_property(self):
        config = [{"type": "div", "x": 255}]
        pipeline = PreprocessingPipeline(config)
        assert pipeline.steps_config == config

    def test_compose_property_single(self):
        config = [{"type": "div", "x": 255}]
        pipeline = PreprocessingPipeline(config)
        assert pipeline.compose is pipeline

    def test_compose_property_multi(self):
        config = {"img": [{"type": "div", "x": 255}]}
        pipeline = PreprocessingPipeline(config)
        compose = pipeline.compose
        assert isinstance(compose, dict)
        assert "img" in compose


class TestPreprocessingRegistry:
    def test_all_types_registered(self):
        expected = [
            "add", "bgr_to_y_channel", "bgr_to_y_channel_uint8",
            "centercrop", "convertcolor", "div", "expanddim",
            "mul", "normalize", "resize", "subtract", "totensor", "transpose",
        ]
        for name in expected:
            assert name in PREPROCESSING_REGISTRY, f"{name} not registered"
