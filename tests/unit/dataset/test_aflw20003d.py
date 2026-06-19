"""Tests for dx_modelzoo.dataset.aflw20003d."""

import pytest

from dx_modelzoo.dataset.aflw20003d import AFLW20003D


class TestAFLW20003D:
    def test_import(self):
        """Verify the class is importable and registered."""
        assert AFLW20003D is not None

    def test_init_with_fake_path(self):
        """Verify __init__ doesn't crash with a fake path (no file access yet)."""
        try:
            ds = AFLW20003D("/nonexistent/path")
            assert ds.data_dir == "/nonexistent/path"
        except (FileNotFoundError, OSError):
            # Some datasets validate path in __init__ — that's fine
            pass

    def test_is_dataset_base(self):
        from dx_modelzoo.common.dataloader import DatasetBase
        assert issubclass(AFLW20003D, DatasetBase)
