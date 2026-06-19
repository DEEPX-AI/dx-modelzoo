"""Tests for dx_modelzoo.common.seed."""

import os
import random

import numpy as np

from dx_modelzoo.common.seed import get_seed, set_seed


class TestSetSeed:
    def test_set_and_get(self):
        set_seed(42)
        assert get_seed() == 42

    def test_numpy_deterministic(self):
        set_seed(123)
        a = np.random.rand(10)
        set_seed(123)
        b = np.random.rand(10)
        np.testing.assert_array_equal(a, b)

    def test_python_random_deterministic(self):
        set_seed(99)
        a = [random.random() for _ in range(5)]
        set_seed(99)
        b = [random.random() for _ in range(5)]
        assert a == b

    def test_hash_seed_env_set(self):
        set_seed(7)
        assert os.environ["PYTHONHASHSEED"] == "7"


class TestGetSeed:
    def test_returns_none_initially(self):
        # Reset global state
        import dx_modelzoo.common.seed as mod
        mod._global_seed = None
        assert get_seed() is None
