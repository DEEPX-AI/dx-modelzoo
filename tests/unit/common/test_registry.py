"""Tests for dx_modelzoo.common.registry."""

import pytest

from dx_modelzoo.common.registry import Registry, _derive_name


class TestDeriveName:
    def test_strips_dataset_suffix(self):
        class BSD68Dataset:
            pass
        assert _derive_name(BSD68Dataset) == "BSD68"

    def test_no_suffix(self):
        class MyClass:
            pass
        assert _derive_name(MyClass) == "MyClass"

    def test_only_suffix_not_stripped(self):
        class Dataset:
            pass
        # "Dataset" alone should not be stripped to empty
        assert _derive_name(Dataset) == "Dataset"


class TestRegistry:
    def test_register_explicit_name(self):
        reg = Registry("test")

        @reg.register("foo")
        class Foo:
            pass

        assert reg.get("foo") is Foo

    def test_register_auto_derive(self):
        reg = Registry("test")

        @reg.register
        class BarDataset:
            pass

        assert reg.get("Bar") is BarDataset

    def test_register_with_none(self):
        reg = Registry("test")

        @reg.register()
        class BazDataset:
            pass

        assert reg.get("Baz") is BazDataset

    def test_duplicate_raises(self):
        reg = Registry("test")

        @reg.register("dup")
        class A:
            pass

        with pytest.raises(ValueError, match="already registered"):
            @reg.register("dup")
            class B:
                pass

    def test_get_missing_raises(self):
        reg = Registry("test")
        with pytest.raises(KeyError, match="not found"):
            reg.get("nonexistent")

    def test_list(self):
        reg = Registry("test")

        @reg.register("a")
        class A:
            pass

        @reg.register("b")
        class B:
            pass

        assert sorted(reg.list()) == ["a", "b"]

    def test_contains(self):
        reg = Registry("test")

        @reg.register("x")
        class X:
            pass

        assert "x" in reg
        assert "y" not in reg

    def test_len(self):
        reg = Registry("test")
        assert len(reg) == 0

        @reg.register("item")
        class Item:
            pass

        assert len(reg) == 1

    def test_registry_key_stored_on_class(self):
        reg = Registry("test")

        @reg.register("mykey")
        class C:
            pass

        assert C.__registry_key__ == "mykey"
