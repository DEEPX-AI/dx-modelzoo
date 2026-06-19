"""Tests for dx_modelzoo.postprocessing (PostprocessingPipeline + registry)."""

import numpy as np
import pytest

from dx_modelzoo.postprocessing import POSTPROCESSING_REGISTRY, PostprocessingPipeline


class TestPostprocessingRegistry:
    def test_topk_registered(self):
        assert "topk" in POSTPROCESSING_REGISTRY

    def test_identity_registered(self):
        assert "identity" in POSTPROCESSING_REGISTRY

    def test_nms_registered(self):
        assert "nms" in POSTPROCESSING_REGISTRY

    def test_segmentation_argmax_registered(self):
        assert "segmentation_argmax" in POSTPROCESSING_REGISTRY


class TestPostprocessingPipeline:
    def test_empty_pipeline_passthrough(self):
        pipeline = PostprocessingPipeline([])
        data = [np.array([1, 2, 3])]
        result = pipeline(data)
        assert result is data

    def test_single_identity_step(self):
        pipeline = PostprocessingPipeline([{"type": "identity"}])
        data = [np.array([1, 2, 3])]
        result = pipeline(data)
        np.testing.assert_array_equal(result, np.array([1, 2, 3]))

    def test_missing_type_raises(self):
        with pytest.raises(ValueError, match="missing 'type'"):
            PostprocessingPipeline([{"k": [1, 5]}])

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="unknown type"):
            PostprocessingPipeline([{"type": "nonexistent_step"}])

    def test_invalid_params_raises(self):
        with pytest.raises(ValueError, match="invalid params"):
            PostprocessingPipeline([{"type": "topk", "invalid_param": 999}])

    def test_topk_pipeline(self):
        pipeline = PostprocessingPipeline([{"type": "topk", "k": [1, 3]}])
        logits = np.array([[0.1, 0.9, 0.5, 0.3]])
        result = pipeline([logits])
        assert result.shape[-1] == 3  # top-3 indices
