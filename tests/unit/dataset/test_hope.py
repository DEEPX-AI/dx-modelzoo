"""Tests for dx_modelzoo.dataset.hope."""

import pytest

from dx_modelzoo.dataset.hope import HOPE


class TestHOPE:
    def test_import(self):
        """Verify the class is importable."""
        assert HOPE is not None

    def test_is_dataset_base(self):
        from dx_modelzoo.common.dataloader import DatasetBase
        assert issubclass(HOPE, DatasetBase)
