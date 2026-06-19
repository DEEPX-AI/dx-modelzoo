"""Tests for dx_modelzoo.loader.config."""

import os

from dx_modelzoo.loader.config import resolve_variables, resolve_variables_recursive


class TestResolveVariables:
    def test_env_var(self, monkeypatch):
        monkeypatch.setenv("TEST_VAR", "/data/path")
        result = resolve_variables("${TEST_VAR}/images", "test.yaml")
        assert result == "/data/path/images"

    def test_missing_var_left_asis(self, monkeypatch):
        monkeypatch.delenv("MISSING_VAR", raising=False)
        result = resolve_variables("${MISSING_VAR}/path", "test.yaml")
        assert result == "${MISSING_VAR}/path"

    def test_extra_vars_take_precedence(self, monkeypatch):
        monkeypatch.setenv("MY_VAR", "env_value")
        result = resolve_variables("${MY_VAR}", "test.yaml", extra_vars={"MY_VAR": "extra_value"})
        assert result == "extra_value"

    def test_no_vars(self):
        result = resolve_variables("plain text", "test.yaml")
        assert result == "plain text"

    def test_multiple_vars(self, monkeypatch):
        monkeypatch.setenv("A", "hello")
        monkeypatch.setenv("B", "world")
        result = resolve_variables("${A}_${B}", "test.yaml")
        assert result == "hello_world"


class TestResolveVariablesRecursive:
    def test_dict_with_name(self, monkeypatch):
        monkeypatch.setenv("MODEL_ROOT", "/models")
        data = {"name": "resnet50", "path": "${MODEL_ROOT}/${MODEL_NAME}"}
        result = resolve_variables_recursive(data, "test.yaml")
        assert result["path"] == "/models/resnet50"

    def test_list(self, monkeypatch):
        monkeypatch.setenv("X", "val")
        data = ["${X}", "plain"]
        result = resolve_variables_recursive(data, "test.yaml")
        assert result == ["val", "plain"]

    def test_non_string_passthrough(self):
        assert resolve_variables_recursive(42, "test.yaml") == 42
        assert resolve_variables_recursive(None, "test.yaml") is None

    def test_nested_dict(self, monkeypatch):
        monkeypatch.setenv("ROOT", "/data")
        data = {"name": "model", "dataset": {"path": "${ROOT}/set"}}
        result = resolve_variables_recursive(data, "test.yaml")
        assert result["dataset"]["path"] == "/data/set"
