"""Tests for dx_modelzoo.postprocessing.box_decoder."""

import numpy as np

from dx_modelzoo.postprocessing.box_decoder import generate_ssd_priors


class TestGenerateSsdPriors:
    def test_output_shape(self):
        priors = generate_ssd_priors(image_size=(640, 640))
        assert priors.ndim == 2
        assert priors.shape[1] == 4

    def test_values_normalized(self):
        priors = generate_ssd_priors(image_size=(320, 320))
        # Priors should be in [0, 1] range (normalized coords)
        assert priors[:, :2].min() >= 0.0
        assert priors[:, :2].max() <= 1.0

    def test_custom_steps(self):
        priors_a = generate_ssd_priors(image_size=(640, 640), steps=(8, 16, 32))
        priors_b = generate_ssd_priors(image_size=(640, 640), steps=(8, 16))
        # Fewer steps = fewer priors
        assert priors_b.shape[0] < priors_a.shape[0]
