"""Tests for dx_modelzoo.common.dataloader."""

import numpy as np
import pytest

from dx_modelzoo.common.dataloader import DataLoader, DatasetBase, _numpy_collate, make_dataloader


class SimpleDataset(DatasetBase):
    """Minimal concrete dataset for testing."""

    def __init__(self, size=10):
        super().__init__("/fake/path")
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img = np.ones((3, 32, 32), dtype=np.float32) * idx
        label = idx % 5
        return img, label

    def validate(self):
        return super().validate()


class MappingDataset(DatasetBase):
    def __init__(self, size=3):
        super().__init__("/fake/path")
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {"image": np.array([idx], dtype=np.float32), "tokens": np.array([idx + 1], dtype=np.int64)}

    def validate(self):
        pass


class TestDatasetBase:
    def test_abstract_methods(self):
        with pytest.raises(TypeError):
            DatasetBase("/path")

    def test_preprocessing_not_set_raises(self):
        ds = SimpleDataset()
        with pytest.raises(ValueError, match="not set"):
            _ = ds.preprocessing

    def test_ensure_exists_raises_on_missing(self, tmp_path):
        ds = SimpleDataset()
        fake_dir = str(tmp_path / "nonexistent")
        with pytest.raises(FileNotFoundError, match="Dataset not found"):
            ds.ensure_exists(fake_dir, "Install instructions here")

    def test_ensure_exists_passes_on_existing(self, tmp_path):
        ds = SimpleDataset()
        existing = tmp_path / "data"
        existing.mkdir()
        ds.ensure_exists(str(existing), "guide")  # Should not raise


class TestNumpyCollate:
    def test_empty_batch(self):
        assert _numpy_collate([]) == []

    def test_tuple_batch(self):
        batch = [
            (np.array([1.0, 2.0]), np.array([3.0])),
            (np.array([4.0, 5.0]), np.array([6.0])),
        ]
        result = _numpy_collate(batch)
        assert len(result) == 2
        np.testing.assert_array_equal(result[0], np.array([[1.0, 2.0], [4.0, 5.0]]))

    def test_ndarray_batch(self):
        batch = [np.array([1, 2]), np.array([3, 4])]
        result = _numpy_collate(batch)
        assert result.shape == (2, 2)

    def test_non_array_batch(self):
        batch = [1, 2, 3]
        assert _numpy_collate(batch) == [1, 2, 3]

    def test_mapping_batch_preserves_named_inputs(self):
        result = _numpy_collate(
            [
                {"image": np.array([1]), "tokens": np.array([2])},
                {"image": np.array([3]), "tokens": np.array([4])},
            ]
        )
        assert set(result) == {"image", "tokens"}
        np.testing.assert_array_equal(result["image"], [[1], [3]])


class TestDataLoader:
    def test_len_single_batch(self):
        ds = SimpleDataset(10)
        dl = DataLoader(ds, batch_size=1)
        assert len(dl) == 10

    def test_len_multi_batch(self):
        ds = SimpleDataset(10)
        dl = DataLoader(ds, batch_size=3)
        assert len(dl) == 4  # ceil(10/3)

    def test_iter_single_process(self):
        ds = SimpleDataset(5)
        dl = DataLoader(ds, batch_size=1, num_workers=0)
        items = list(dl)
        assert len(items) == 5

    def test_iter_batch_size_2(self):
        ds = SimpleDataset(5)
        dl = DataLoader(ds, batch_size=2, num_workers=0)
        items = list(dl)
        assert len(items) == 3  # 2+2+1

    def test_shuffle_changes_order(self):
        ds = SimpleDataset(100)
        dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)
        items = list(dl)
        # With 100 items, shuffle should produce different order
        labels = [item[1] for item in items]
        assert labels != list(range(100))  # Extremely unlikely to be sorted

    def test_custom_collate_fn(self):
        ds = SimpleDataset(4)
        dl = DataLoader(ds, batch_size=2, collate_fn=lambda x: len(x))
        items = list(dl)
        assert items == [2, 2]

    def test_mapping_batch_size_one_keeps_batch_dimension(self):
        item = next(iter(DataLoader(MappingDataset(), batch_size=1)))
        assert item["image"].shape == (1, 1)
        assert item["tokens"].shape == (1, 1)

    def test_start_method_rejects_unknown_value(self):
        with pytest.raises(ValueError, match="Unsupported multiprocessing start method"):
            DataLoader(MappingDataset(), num_workers=1, start_method="invalid")

    @pytest.mark.parametrize("start_method", ["forkserver", "spawn"])
    def test_mapping_workers_preserve_order(self, start_method):
        import multiprocessing as mp

        if start_method not in mp.get_all_start_methods():
            pytest.skip(f"{start_method} unavailable")
        batches = list(DataLoader(MappingDataset(4), batch_size=1, num_workers=1, start_method=start_method))
        assert [int(batch["image"][0, 0]) for batch in batches] == [0, 1, 2, 3]


class TestMakeDataloader:
    def test_returns_dataloader(self):
        ds = SimpleDataset(5)
        dl = make_dataloader(ds, batch_size=2)
        assert isinstance(dl, DataLoader)
        assert len(dl) == 3
