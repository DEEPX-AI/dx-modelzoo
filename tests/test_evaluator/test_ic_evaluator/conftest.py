import pytest
import torch
from torch.utils.data import Dataset


@pytest.fixture
def dummy_ic_dataset():
    class Dummydataset(Dataset):
        def __init__(self, size):
            self.size = size
            self.fake_images = [torch.randn(3, 224, 224) for _ in range(self.size)]

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return self.fake_images[idx], idx

    return Dummydataset(8)


@pytest.fixture
def dummy_session():
    class DummySession:
        def run(self, *args, **kwargs):
            return [torch.ones(1, 10).numpy()]

    return DummySession()
