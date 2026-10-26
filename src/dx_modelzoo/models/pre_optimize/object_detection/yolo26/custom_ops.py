"""YOLO26 object detection output decoding."""
from __future__ import annotations

from dx_modelzoo.postprocessing import POSTPROCESSING_REGISTRY
from dx_modelzoo.postprocessing.coord_scaler import unpad_and_scale


@POSTPROCESSING_REGISTRY.register("yolo26_decode")
class YOLO26Decode:
    def __init__(self, conf_thres: float = 0.001, max_output_boxes: int = 300, pad_resize: bool = True, **kwargs):
        self.conf_thres = conf_thres
        self.max_output_boxes = max_output_boxes
        self.pad_resize = pad_resize

    def __call__(self, outputs, **kwargs):
        out = outputs[0] if isinstance(outputs, list) else outputs
        if out.ndim == 3:
            out = out[0]
        if out.shape[0] > 0 and out.shape[1] >= 5:
            mask = out[:, 4] > self.conf_thres
            out = out[mask]
        if out.shape[0] > self.max_output_boxes:
            out = out[: self.max_output_boxes]

        # Rescale xyxy box coordinates to original image space
        origin_hw = kwargs.get("origin_hw")
        input_hw = kwargs.get("input_hw")
        if origin_hw is not None and input_hw is not None and len(out) > 0:
            out = out.copy()
            out[:, :4] = unpad_and_scale(
                out[:, :4],
                input_hw,
                origin_hw,
                pad_resize=self.pad_resize,
            )
        return out
