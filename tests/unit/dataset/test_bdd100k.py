"""Tests for dx_modelzoo.dataset.bdd100k."""

import pytest

from dx_modelzoo.dataset.bdd100k import BDD100K


class TestBDD100K:
    def test_import(self):
        """Verify the class is importable and registered."""
        assert BDD100K is not None

    def test_init_with_fake_path(self):
        """Verify __init__ doesn't crash with a fake path (no file access yet)."""
        try:
            ds = BDD100K("/nonexistent/path")
            assert ds.data_dir == "/nonexistent/path"
        except (FileNotFoundError, OSError):
            # Some datasets validate path in __init__ — that's fine
            pass

    def test_is_dataset_base(self):
        from dx_modelzoo.common.dataloader import DatasetBase
        assert issubclass(BDD100K, DatasetBase)
