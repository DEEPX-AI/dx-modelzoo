from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, List, Union

import numpy as np
from torchvision.transforms import Compose

from dx_modelzoo.common.registry import Registry

__all__ = ["PREPROCESSING_REGISTRY", "PreprocessingPipeline"]

PREPROCESSING_REGISTRY = Registry("preprocessing")
NPU_SKIP_DEFAULT = {"div", "normalize", "transpose", "expanddim", "mul", "add", "subtract"}


class PreprocessingPipeline(Compose):
    """Ordered pipeline of preprocessing operations, built from YAML config.

    Inherits from ``torchvision.transforms.Compose`` so that dx_com's
    auto-extraction (``_is_compose()``) recognizes it directly.
    The ``.transforms`` attribute contains the fusable ops (ToTensor, Normalize)
    for preprocessing fusion.  The pipeline itself executes numpy-based steps
    via ``__call__``.

    When npu_mode=True, arithmetic/normalization ops (div, normalize, transpose,
    expanddim, mul, add, subtract) are skipped because the NPU handles them
    internally (uint8 input models). For float32 input NPU models, npu_mode
    should be False so all ops are kept.

    Accepts either a flat ``List[dict]`` (single-input) or a
    ``Dict[str, List[dict]]`` (multi-input) config.
    """

    def __init__(
        self,
        steps_config: Union[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]],
        npu_mode: bool = False,
    ) -> None:
        self._raw_config = steps_config

        # Build fusable torchvision transforms for Compose.transforms
        from dx_modelzoo.preprocessing.converter import yaml_to_compose_transforms

        tv_transforms = yaml_to_compose_transforms(steps_config)
        super().__init__(tv_transforms if isinstance(tv_transforms, list) else [])

        self._steps_by_input: Dict[str, List[Any]] = {}
        if isinstance(steps_config, dict):
            for input_name, modal_steps in steps_config.items():
                if isinstance(modal_steps, list):
                    self._steps_by_input[input_name] = self._build_steps(modal_steps, npu_mode)
            self._steps_config = steps_config
            self.steps = [step for steps in self._steps_by_input.values() for step in steps]
        else:
            self._steps_config = steps_config
            self.steps = self._build_steps(steps_config, npu_mode)

    @staticmethod
    def _build_steps(steps_config, npu_mode):
        result = []
        for step_cfg in steps_config:
            step_type = step_cfg["type"]
            if npu_mode and step_type in NPU_SKIP_DEFAULT:
                continue
            cls = PREPROCESSING_REGISTRY.get(step_type)
            params = {k: v for k, v in step_cfg.items() if k != "type"}
            result.append(cls(**params))
        return result

    @property
    def steps_config(self) -> Union[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
        """Original YAML step dicts used to build this pipeline."""
        return self._steps_config

    @property
    def compose(self) -> Any:
        """Backward-compatible property.  Returns self (single-input) or
        Dict[str, Compose] (multi-input)."""
        if isinstance(self._raw_config, dict):
            from dx_modelzoo.preprocessing.converter import yaml_to_compose_multi_dict

            return yaml_to_compose_multi_dict(self._raw_config)
        return self

    def __call__(self, inputs: Union[np.ndarray, Mapping]) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if self._steps_by_input:
            if not isinstance(inputs, Mapping):
                raise TypeError("Multi-input preprocessing requires mapping input")
            expected = set(self._steps_by_input)
            provided = set(inputs)
            if provided != expected:
                raise ValueError(
                    f"Preprocessing input keys mismatch. Missing: {sorted(expected - provided)}; "
                    f"unknown: {sorted(provided - expected)}"
                )
            return {name: self._apply_steps(inputs[name], self._steps_by_input[name]) for name in self._steps_by_input}
        return self._apply_steps(inputs, self.steps)

    @staticmethod
    def _apply_steps(inputs, steps):
        for step in steps:
            inputs = step(inputs)
        return inputs


# Import submodules to trigger registration
from dx_modelzoo.preprocessing import (  # noqa: F401, E402
    add,
    bgr_to_y_channel,
    bgr_to_y_channel_uint8,
    centercrop,
    convertcolor,
    div,
    expanddim,
    mul,
    normalize,
    resize,
    subtract,
    totensor,
    transpose,
)
