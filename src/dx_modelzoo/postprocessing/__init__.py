"""Postprocessing pipeline and registry.

Coordinate Space Convention
---------------------------
Each postprocessor rescales its output from model-input pixel space
to original image coordinates using ``coord_scaler.unpad_and_scale``.
The evaluator passes ``origin_hw`` and ``input_hw`` via kwargs;
postprocessors that produce coordinate outputs (boxes, keypoints)
apply the inverse of the preprocessing resize/pad transformation.

Detection Output Format
-----------------------
All detection postprocessors return ``np.ndarray [M, 6]``
with columns ``(x1, y1, x2, y2, score, class_id)``.
"""
from __future__ import annotations

from typing import Any, Dict, List

from dx_modelzoo.common.registry import Registry

__all__ = ["POSTPROCESSING_REGISTRY", "PostprocessingPipeline"]

POSTPROCESSING_REGISTRY = Registry("postprocessing")


class PostprocessingPipeline:
    """Ordered pipeline of postprocessing operations."""

    def __init__(self, steps_config: List[Dict[str, Any]]) -> None:
        self.steps = []
        for i, step_cfg in enumerate(steps_config):
            step_type = step_cfg.get("type")
            if step_type is None:
                raise ValueError(f"Postprocessing step {i}: missing 'type' key")
            if step_type not in POSTPROCESSING_REGISTRY:
                available = ", ".join(sorted(POSTPROCESSING_REGISTRY.list()))
                raise ValueError(f"Postprocessing step {i}: unknown type '{step_type}'. " f"Available: {available}")
            cls = POSTPROCESSING_REGISTRY.get(step_type)
            params = {k: v for k, v in step_cfg.items() if k != "type"}
            try:
                self.steps.append(cls(**params))
            except TypeError as e:
                raise ValueError(f"Postprocessing step {i} (type='{step_type}'): " f"invalid params — {e}") from e

    def __call__(self, outputs, **kwargs):
        if not self.steps:
            return outputs
        result = outputs
        for step in self.steps:
            result = step(result, **kwargs)
        return result


from dx_modelzoo.postprocessing import identity, masks, nms, segmentation_argmax, topk  # noqa: F401, E402
