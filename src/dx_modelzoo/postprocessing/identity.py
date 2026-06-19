from __future__ import annotations

from dx_modelzoo.postprocessing import POSTPROCESSING_REGISTRY


@POSTPROCESSING_REGISTRY.register("identity")
class Identity:
    """Pass-through: unwrap single-element list, otherwise return as-is."""

    def __call__(self, outputs, **kwargs):
        if isinstance(outputs, list) and len(outputs) == 1:
            return outputs[0]
        return outputs
