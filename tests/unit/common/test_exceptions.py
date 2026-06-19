"""Tests for dx_modelzoo.common.exceptions."""

import pytest

from dx_modelzoo.common.exceptions import ConfigError, ModelBuildError, YamlValidationError


class TestExceptions:
    def test_config_error_is_exception(self):
        with pytest.raises(ConfigError):
            raise ConfigError("bad config")

    def test_yaml_validation_error_is_exception(self):
        with pytest.raises(YamlValidationError):
            raise YamlValidationError("invalid yaml")

    def test_model_build_error_is_exception(self):
        with pytest.raises(ModelBuildError):
            raise ModelBuildError("build failed")

    def test_message_preserved(self):
        err = ConfigError("test message")
        assert str(err) == "test message"
