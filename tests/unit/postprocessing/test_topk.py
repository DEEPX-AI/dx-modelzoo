"""Tests for dx_modelzoo.postprocessing.topk."""

import numpy as np

from dx_modelzoo.postprocessing.topk import TopK


class TestTopK:
    def test_basic_topk(self):
        op = TopK(k=[1, 5])
        logits = np.array([0.1, 0.9, 0.5, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 0.05])
        result = op([logits])
        assert result.shape[-1] == 5  # max(k)
        assert result[0] == 1  # highest score index

    def test_batch_input(self):
        op = TopK(k=[1, 3])
        logits = np.array([[0.1, 0.9, 0.5], [0.8, 0.2, 0.6]])
        result = op([logits])
        assert result.shape == (2, 3)

    def test_skip_background(self):
        op = TopK(k=[1], skip_background=True)
        logits = np.array([0.99, 0.1, 0.9])  # index 0 is background
        result = op([logits])
        # After removing background (index 0), we have [0.1, 0.9] → top1 = index 1
        assert result[0] == 1

    def test_default_k(self):
        op = TopK()
        assert op.k == [1, 5]

    def test_list_output_unwrap(self):
        op = TopK(k=[1])
        logits = np.array([0.3, 0.7, 0.1])
        result = op([logits])
        assert result[0] == 1
