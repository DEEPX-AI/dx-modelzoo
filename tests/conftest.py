"""Top-level conftest for dx-modelzoo tests."""

import pytest
import numpy as np


@pytest.fixture
def sample_image_chw():
    """Sample CHW float32 image (3, 224, 224)."""
    return np.random.rand(3, 224, 224).astype(np.float32)


@pytest.fixture
def sample_image_hwc():
    """Sample HWC uint8 image (224, 224, 3)."""
    return (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
