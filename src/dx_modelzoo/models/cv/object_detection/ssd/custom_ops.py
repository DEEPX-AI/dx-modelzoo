"""SSD per-class hard NMS postprocessor."""
from __future__ import annotations

import numpy as np

from dx_modelzoo.postprocessing import POSTPROCESSING_REGISTRY
from dx_modelzoo.postprocessing.coord_scaler import unpad_and_scale
from dx_modelzoo.postprocessing.nms import _hard_nms as hard_nms


@POSTPROCESSING_REGISTRY.register("ssd_postprocess")
class SSDPostprocess:
    """SSD per-class hard NMS. Includes self-contained NMS.

    Model outputs: [scores(B,N,C), boxes(B,N,4)] or reversed.
    Returns: np.ndarray [M, 6] (x1, y1, x2, y2, score, class_id)
    """

    def __init__(
        self,
        conf_thres: float = 0.01,
        iou_thres: float = 0.45,
        variant: str = "ssd",
        input_size: int = 300,
        pad_resize: bool = False,
        **kwargs,
    ):
        self.conf_thres = conf_thres
        self.pad_resize = pad_resize
        self.iou_thres = iou_thres
        self.variant = variant
        self.input_size = input_size

    def _to_pixel_boxes(self, boxes: np.ndarray) -> np.ndarray:
        boxes = boxes.astype(np.float32, copy=True)
        if boxes.size == 0:
            return boxes
        if np.max(np.abs(boxes)) <= 2.0:
            boxes = np.clip(boxes, 0.0, 1.0) * float(self.input_size)
        return boxes

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
        if not isinstance(outputs, list) or len(outputs) < 2:
            return np.empty((0, 6), dtype=np.float64)

        # Identify scores vs boxes by last dimension
        if outputs[0].shape[-1] == 4:
            boxes_out, scores_out = outputs[0], outputs[1]
        else:
            scores_out, boxes_out = outputs[0], outputs[1]

        if scores_out.ndim == 2:
            scores_out = scores_out[np.newaxis, ...]
        if boxes_out.ndim == 2:
            boxes_out = boxes_out[np.newaxis, ...]

        num_classes = scores_out.shape[2]
        all_results = []

        for class_scores, box_coords in zip(scores_out, boxes_out):
            picked_boxes, picked_scores, picked_classes = [], [], []

            for cls_idx in range(1, num_classes):
                mask = class_scores[:, cls_idx] > self.conf_thres
                filt_scores = class_scores[mask, cls_idx]
                filt_boxes = box_coords[mask, :]

                if filt_boxes.shape[0] == 0:
                    continue

                pixel_boxes = self._to_pixel_boxes(filt_boxes)
                keep = hard_nms(pixel_boxes, filt_scores, self.iou_thres)
                picked_boxes.append(pixel_boxes[keep])

                picked_scores.append(filt_scores[keep])
                picked_classes.append(np.full(len(keep), cls_idx, dtype=np.float64))

            if picked_boxes:
                result = np.column_stack(
                    [
                        np.concatenate(picked_boxes),
                        np.concatenate(picked_scores),
                        np.concatenate(picked_classes),
                    ]
                )
                all_results.append(result)

        if not all_results:
            return np.empty((0, 6), dtype=np.float64)
        result = np.concatenate(all_results, axis=0).astype(np.float64)
        return self._rescale_result(result, **kwargs)
