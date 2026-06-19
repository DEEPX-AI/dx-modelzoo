"""Tests for dx_modelzoo.preprocessing.converter."""

from torchvision import transforms

from dx_modelzoo.preprocessing.converter import (
    _ArithmeticTransform,
    _extract_fusable_transforms,
    yaml_to_compose,
    yaml_to_compose_multi,
    yaml_to_compose_multi_dict,
    yaml_to_compose_transforms,
)


class TestArithmeticTransform:
    def test_div(self):
        t = _ArithmeticTransform("div", 2.0)
        assert t(10.0) == 5.0

    def test_mul(self):
        t = _ArithmeticTransform("mul", 3.0)
        assert t(4.0) == 12.0

    def test_add(self):
        t = _ArithmeticTransform("add", 1.0)
        assert t(5.0) == 6.0

    def test_sub(self):
        t = _ArithmeticTransform("sub", 2.0)
        assert t(7.0) == 5.0

    def test_unknown_op_passthrough(self):
        t = _ArithmeticTransform("unknown", 1.0)
        assert t(42.0) == 42.0

    def test_repr(self):
        t = _ArithmeticTransform("div", 255.0)
        assert "div" in repr(t)


class TestExtractFusableTransforms:
    def test_div_255_becomes_totensor(self):
        steps = [{"type": "div", "x": 255}]
        result = _extract_fusable_transforms(steps)
        assert len(result) == 1
        assert isinstance(result[0], transforms.ToTensor)

    def test_div_non255_becomes_arithmetic(self):
        steps = [{"type": "div", "x": 2}]
        result = _extract_fusable_transforms(steps)
        assert len(result) == 1
        assert isinstance(result[0], _ArithmeticTransform)

    def test_normalize(self):
        steps = [{"type": "normalize", "mean": [0.5], "std": [0.5]}]
        result = _extract_fusable_transforms(steps)
        assert len(result) == 1
        assert isinstance(result[0], transforms.Normalize)

    def test_non_fusable_skipped(self):
        steps = [{"type": "resize", "size": [224, 224]}, {"type": "centercrop", "height": 100, "width": 100}]
        result = _extract_fusable_transforms(steps)
        assert len(result) == 0

    def test_totensor_step(self):
        steps = [{"type": "totensor"}]
        result = _extract_fusable_transforms(steps)
        assert isinstance(result[0], transforms.ToTensor)

    def test_mul_add_subtract(self):
        steps = [
            {"type": "mul", "x": 2.0},
            {"type": "add", "x": 1.0},
            {"type": "subtract", "x": 0.5},
        ]
        result = _extract_fusable_transforms(steps)
        assert len(result) == 3


class TestYamlToCompose:
    def test_returns_compose(self):
        steps = [{"type": "div", "x": 255}]
        result = yaml_to_compose(steps)
        assert isinstance(result, transforms.Compose)

    def test_empty_steps(self):
        result = yaml_to_compose([])
        assert isinstance(result, transforms.Compose)
        assert len(result.transforms) == 0


class TestYamlToComposeTransforms:
    def test_list_input(self):
        steps = [{"type": "div", "x": 255}]
        result = yaml_to_compose_transforms(steps)
        assert len(result) == 1

    def test_dict_input(self):
        config = {"input0": [{"type": "div", "x": 255}], "input1": [{"type": "normalize", "mean": [0.5], "std": [0.5]}]}
        result = yaml_to_compose_transforms(config)
        assert len(result) == 2

    def test_non_list_non_dict(self):
        result = yaml_to_compose_transforms("invalid")
        assert result == []


class TestYamlToComposeMulti:
    def test_single_input(self):
        steps = [{"type": "div", "x": 255}]
        result = yaml_to_compose_multi(steps)
        assert isinstance(result, transforms.Compose)

    def test_empty_single_returns_none(self):
        result = yaml_to_compose_multi([{"type": "resize", "size": [224, 224]}])
        assert result is None

    def test_dict_input(self):
        config = {"img": [{"type": "div", "x": 255}]}
        result = yaml_to_compose_multi(config)
        assert isinstance(result, dict)
        assert "img" in result

    def test_dict_empty_returns_none(self):
        config = {"img": [{"type": "resize", "size": [224, 224]}]}
        result = yaml_to_compose_multi(config)
        assert result is None

    def test_invalid_type_returns_none(self):
        result = yaml_to_compose_multi(123)
        assert result is None


class TestYamlToComposeMultiDict:
    def test_basic(self):
        config = {"img": [{"type": "div", "x": 255}]}
        result = yaml_to_compose_multi_dict(config)
        assert "img" in result
        assert isinstance(result["img"], transforms.Compose)
