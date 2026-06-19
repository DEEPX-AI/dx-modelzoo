"""Tests for dx_modelzoo.command.create_yaml."""

from dx_modelzoo.command.create_yaml import _can_prompt_interactively, _check_builtin_shadow


class TestCanPromptInteractively:
    def test_returns_bool(self):
        result = _can_prompt_interactively()
        assert isinstance(result, bool)


class TestCheckBuiltinShadow:
    def test_no_shadow_passes(self):
        # A name that certainly doesn't exist as builtin
        _check_builtin_shadow("totally_fake_model_xyz_999")
