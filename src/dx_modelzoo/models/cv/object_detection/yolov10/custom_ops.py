"""YOLOv10 output decoding: passthrough with confidence filtering (NMS-free model)."""
from __future__ import annotations

from dx_modelzoo.postprocessing import POSTPROCESSING_REGISTRY
from dx_modelzoo.postprocessing.coord_scaler import unpad_and_scale


@POSTPROCESSING_REGISTRY.register("yolov10_decode")
class YOLOv10Decode:
    def __init__(self, conf_thres: float = 0.001, max_output_boxes: int = 300, pad_resize: bool = True, **kwargs):
        self.conf_thres = conf_thres
        self.max_output_boxes = max_output_boxes
        self.pad_resize = pad_resize

    def _rescale_result(self, result, **kwargs):
        """Rescale box coordinates from model-input space to original image space."""
        origin_hw = kwargs.get("origin_hw")
        input_hw = kwargs.get("input_hw")
        if origin_hw is None or input_hw is None or len(result) == 0:
            return result
        result = result.copy()
        result[:, :4] = unpad_and_scale(
            result[:, :4],
            input_hw,
            origin_hw,
            pad_resize=getattr(self, "pad_resize", True),
        )
        return result

    def __call__(self, outputs, **kwargs):
        out = outputs[0] if isinstance(outputs, list) else outputs
        if out.ndim == 3:
            out = out[0]
        if out.shape[0] > 0 and out.shape[1] >= 5:
            mask = out[:, 4] > self.conf_thres
            out = out[mask]
        if out.shape[0] > self.max_output_boxes:
            out = out[: self.max_output_boxes]
        return self._rescale_result(out, **kwargs)
