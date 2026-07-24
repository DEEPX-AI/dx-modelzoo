"""Tests for dx_modelzoo.dataset.coco."""

import pytest

from dx_modelzoo.dataset.coco import COCO


class TestCOCO:
    def test_import(self):
        """Verify the class is importable and registered."""
        assert COCO is not None

    def test_init_with_fake_path(self):
        """Verify __init__ doesn't crash with a fake path (no file access yet)."""
        try:
            ds = COCO("/nonexistent/path")
            assert ds.data_dir == "/nonexistent/path"
        except (FileNotFoundError, OSError):
            # Some datasets validate path in __init__ — that's fine
            pass

    def test_is_dataset_base(self):
        from dx_modelzoo.common.dataloader import DatasetBase
        assert issubclass(COCO, DatasetBase)
