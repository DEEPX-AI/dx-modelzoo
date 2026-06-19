"""Tests for dx_modelzoo.common.paths."""

from pathlib import Path

from dx_modelzoo.common.paths import get_builtin_models_dir, get_workspace_custom_dir


class TestGetBuiltinModelsDir:
    def test_returns_path(self):
        result = get_builtin_models_dir()
        assert isinstance(result, Path)

    def test_ends_with_models(self):
        result = get_builtin_models_dir()
        assert result.name == "models"

    def test_parent_is_dx_modelzoo(self):
        result = get_builtin_models_dir()
        assert result.parent.name == "dx_modelzoo"


class TestGetWorkspaceCustomDir:
    def test_returns_path(self):
        result = get_workspace_custom_dir()
        assert isinstance(result, Path)

    def test_ends_with_custom(self):
        result = get_workspace_custom_dir()
        assert result.name == "custom"
