"""Tests for dx_modelzoo.loader.model_scaffold."""

import pytest

from dx_modelzoo.loader.model_scaffold import (
    IDENTIFIER_PATTERN,
    PREPROCESSING_PRESETS,
    POSTPROCESSING_PRESETS,
    InvalidIdentifierError,
    DuplicateCustomModelError,
    _default_resize,
    validate_identifier,
    _validate_non_empty,
    parse_input_shape,
    _profiles_for,
    render_classification_config,
    ClassificationScaffold,
)


class TestIdentifierPattern:
    def test_valid_identifiers(self):
        valids = ["resnet50", "vit-b-p16", "model_v2.0", "A123"]
        for v in valids:
            assert IDENTIFIER_PATTERN.match(v), f"{v} should be valid"

    def test_invalid_identifiers(self):
        invalids = ["", "-start", ".start", "has space", "a/b"]
        for v in invalids:
            assert not IDENTIFIER_PATTERN.match(v), f"{v} should be invalid"


class TestPreprocessingPresets:
    def test_imagenet_preset_exists(self):
        assert "imagenet" in PREPROCESSING_PRESETS

    def test_imagenet_produces_steps(self):
        shape = [1, 3, 224, 224]
        steps = PREPROCESSING_PRESETS["imagenet"](shape)
        assert isinstance(steps, list)
        types = [s["type"] for s in steps]
        assert "resize" in types
        assert "normalize" in types
        assert "div" in types


class TestPostprocessingPresets:
    def test_topk_preset(self):
        assert "topk" in POSTPROCESSING_PRESETS
        steps = POSTPROCESSING_PRESETS["topk"]
        assert steps[0]["type"] == "topk"


class TestExceptions:
    def test_invalid_identifier_error(self):
        with pytest.raises(InvalidIdentifierError):
            raise InvalidIdentifierError("bad name")

    def test_duplicate_custom_model_error(self):
        with pytest.raises(DuplicateCustomModelError):
            raise DuplicateCustomModelError("already exists")


class TestDefaultResize:
    def test_224_shape(self):
        result = _default_resize([1, 3, 224, 224])
        assert isinstance(result, int)
        assert result > 224

    def test_256_shape(self):
        result = _default_resize([1, 3, 256, 256])
        assert result > 256


class TestValidateIdentifier:
    def test_valid(self):
        result = validate_identifier("resnet50", "model")
        assert result == "resnet50"

    def test_invalid_raises(self):
        with pytest.raises(InvalidIdentifierError):
            validate_identifier("-invalid", "model")

    def test_empty_raises(self):
        with pytest.raises(InvalidIdentifierError):
            validate_identifier("", "model")


class TestValidateNonEmpty:
    def test_valid(self):
        result = _validate_non_empty("hello", "field")
        assert result == "hello"

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            _validate_non_empty("", "field")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError):
            _validate_non_empty("   ", "field")


class TestParseInputShape:
    def test_basic(self):
        result = parse_input_shape("1,3,224,224")
        assert result == [1, 3, 224, 224]

    def test_with_spaces(self):
        result = parse_input_shape("1, 3, 224, 224")
        assert result == [1, 3, 224, 224]

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            parse_input_shape("[1,3,256,256]")


class TestProfilesFor:
    def test_onnx_only(self):
        result = _profiles_for("onnx only")
        assert "onnx" in result
        assert "q-lite" not in result

    def test_onnx_plus_qlite(self):
        result = _profiles_for("onnx + q-lite")
        assert "onnx" in result
        assert "q-lite" in result
