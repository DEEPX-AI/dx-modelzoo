"""Tests for dx_modelzoo.dataset.synthetic."""

import pytest

from dx_modelzoo.dataset.synthetic import SyntheticMultiInput


class TestSyntheticMultiInput:
    def test_import(self):
        """Verify the class is importable."""
        assert SyntheticMultiInput is not None

    def test_is_dataset_base(self):
        from dx_modelzoo.common.dataloader import DatasetBase
        assert issubclass(SyntheticMultiInput, DatasetBase)
