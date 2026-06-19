"""Convert dx-modelzoo YAML preprocessing config to torchvision Compose.

The YAML preprocessing config contains step dicts like::

    [{"type": "resize", "size": 256},
     {"type": "div", "x": 255},
     {"type": "normalize", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}]

This module extracts the *fusable* steps (``div``, ``normalize``) and converts
them into a ``torchvision.transforms.Compose`` so that ``dx_com``'s
auto-extraction parser can pick them up from ``dataset.preprocessing``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from torchvision import transforms


class _ArithmeticTransform:
    """Picklable arithmetic transform (replacement for unpicklable Lambda)."""

    def __init__(self, op: str, value: float) -> None:
        self.op = op
        self.value = value

    def __call__(self, img):
        if self.op == "div":
            return img / self.value
        elif self.op == "mul":
            return img * self.value
        elif self.op == "add":
            return img + self.value
        elif self.op == "sub":
            return img - self.value
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(op={self.op!r}, value={self.value})"


def _extract_fusable_transforms(yaml_steps: List[Dict[str, Any]]) -> List:
    """Extract fusable torchvision transforms from YAML steps.

    Returns a list of torchvision transform objects (may be empty).

    Fusable ops (handled by dx_com preprocessing fusion):
    - ``div``       → ``ToTensor()`` when x=255, arithmetic transform otherwise
    - ``mul``       → arithmetic transform
    - ``add``       → arithmetic transform
    - ``subtract``  → arithmetic transform
    - ``normalize`` → ``Normalize(mean, std)``
    """
    tv_transforms: list = []

    for step in yaml_steps:
        step_type = step.get("type", "")

        if step_type == "totensor":
            tv_transforms.append(transforms.ToTensor())

        elif step_type == "div":
            x = step.get("x")
            if x == 255 or x == 255.0:
                tv_transforms.append(transforms.ToTensor())
            elif x is not None:
                tv_transforms.append(_ArithmeticTransform("div", float(x)))

        elif step_type == "mul":
            x = step.get("x")
            if x is not None:
                tv_transforms.append(_ArithmeticTransform("mul", float(x)))

        elif step_type == "add":
            x = step.get("x")
            if x is not None:
                tv_transforms.append(_ArithmeticTransform("add", float(x)))

        elif step_type == "subtract":
            x = step.get("x")
            if x is not None:
                tv_transforms.append(_ArithmeticTransform("sub", float(x)))

        elif step_type == "normalize":
            mean = step.get("mean")
            std = step.get("std")
            if mean is not None and std is not None:
                tv_transforms.append(transforms.Normalize(mean=list(mean), std=list(std)))

    return tv_transforms


def yaml_to_compose_transforms(
    preprocessing_config: Union[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]],
) -> List:
    """Extract fusable transforms as a flat list (for Compose.__init__).

    For dict configs, flattens all inputs' fusable transforms into one list.
    Used internally by PreprocessingPipeline(Compose) to populate .transforms.
    """
    if isinstance(preprocessing_config, list):
        return _extract_fusable_transforms(preprocessing_config)

    if isinstance(preprocessing_config, dict):
        result: list = []
        for steps in preprocessing_config.values():
            if isinstance(steps, list):
                result.extend(_extract_fusable_transforms(steps))
        return result

    return []


def yaml_to_compose(yaml_steps: List[Dict[str, Any]]) -> transforms.Compose:
    """Convert YAML preprocessing steps to a torchvision ``Compose``.

    Only fusable arithmetic operations are converted:
    - ``{"type": "div", "x": 255}`` → ``transforms.ToTensor()`` (non-255: ``Lambda``)
    - ``{"type": "normalize", "mean": [...], "std": [...]}`` → ``transforms.Normalize(...)``

    Non-fusable steps (resize, centercrop, convertcolor, transpose, expanddim,
    etc.) are silently skipped.

    Args:
        yaml_steps: List of preprocessing step dicts from YAML config.

    Returns:
        ``Compose`` containing the fusable transforms (may be empty).
    """
    return transforms.Compose(_extract_fusable_transforms(yaml_steps))


def yaml_to_compose_multi_dict(
    preprocessing_config: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, transforms.Compose]:
    """Convert dict YAML config to Dict[str, Compose] (multi-input models).

    Args:
        preprocessing_config: Dict mapping input names to step lists.

    Returns:
        Dict mapping each input name to its Compose (may contain empty Composes).
    """
    result: Dict[str, transforms.Compose] = {}
    for modal_name, steps in preprocessing_config.items():
        if isinstance(steps, list):
            result[modal_name] = yaml_to_compose(steps)
    return result


def yaml_to_compose_multi(
    preprocessing_config: Union[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]],
) -> Optional[Union[transforms.Compose, Dict[str, transforms.Compose]]]:
    """Convert YAML preprocessing config (single or multimodal) to Compose(s).

    Args:
        preprocessing_config: Either a flat list of step dicts (single-input) or
            a dict mapping modality/input names to step lists (multi-input).

    Returns:
        - ``Compose`` for single-input models.
        - ``Dict[str, Compose]`` for multi-input models (keyed by input name).
        - ``None`` if no fusable steps found anywhere.
    """
    if isinstance(preprocessing_config, list):
        comp = yaml_to_compose(preprocessing_config)
        return comp if comp.transforms else None

    if isinstance(preprocessing_config, dict):
        result: Dict[str, transforms.Compose] = {}
        for modal_name, steps in preprocessing_config.items():
            if isinstance(steps, list):
                comp = yaml_to_compose(steps)
                if comp.transforms:
                    result[modal_name] = comp
        return result if result else None

    return None
