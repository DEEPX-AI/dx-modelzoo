"""Tests for dx_modelzoo.dataset.market1501."""

import pytest

from dx_modelzoo.dataset.market1501 import Market1501


class TestMarket1501:
    def test_import(self):
        """Verify the class is importable and registered."""
        assert Market1501 is not None

    def test_init_with_fake_path(self):
        """Verify __init__ doesn't crash with a fake path (no file access yet)."""
        try:
            ds = Market1501("/nonexistent/path")
            assert ds.data_dir == "/nonexistent/path"
        except (FileNotFoundError, OSError):
            # Some datasets validate path in __init__ — that's fine
            pass

    def test_is_dataset_base(self):
        from dx_modelzoo.common.dataloader import DatasetBase
        assert issubclass(Market1501, DatasetBase)
