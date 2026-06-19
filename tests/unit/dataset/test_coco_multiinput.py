"""Tests for dx_modelzoo.dataset.coco_multiinput."""

import pytest

from dx_modelzoo.dataset.coco_multiinput import COCOMultiInput


class TestCOCOMultiInput:
    def test_import(self):
        """Verify the class is importable."""
        assert COCOMultiInput is not None

    def test_is_dataset_base(self):
        from dx_modelzoo.common.dataloader import DatasetBase
        assert issubclass(COCOMultiInput, DatasetBase)
