"""SCRFD postprocessing custom ops."""
from __future__ import annotations

import numpy as np

from dx_modelzoo.postprocessing import POSTPROCESSING_REGISTRY
from dx_modelzoo.postprocessing.box_decoder import generate_scrfd_grids
from dx_modelzoo.postprocessing.coord_scaler import unpad_and_scale
from dx_modelzoo.postprocessing.nms import nms_numpy


@POSTPROCESSING_REGISTRY.register("scrfd_decode")
class SCRFDDecode:
    """SCRFD anchor-free face detection decoder."""

    def __init__(
        self,
        conf_thres: float = 0.02,
        iou_thres: float = 0.45,
        **kwargs,
    ) -> None:
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def __call__(self, outputs, origin_hw=None, input_hw=None, **kwargs):
        if input_hw is None:
            input_hw = origin_hw
        if origin_hw is None:
            origin_hw = input_hw

        h_model, w_model = input_hw
        grids, strides_arr = generate_scrfd_grids(h_model, w_model)
        if grids.shape[0] == 0:
            return []

        conf_outs = sorted(
            [o for o in outputs if o.shape[-1] == 1],
            key=lambda x: x.shape[1],
            reverse=True,
        )
        box_outs = sorted(
            [o for o in outputs if o.shape[-1] == 4],
            key=lambda x: x.shape[1],
            reverse=True,
        )
        if not conf_outs or not box_outs:
            return []

        conf = np.concatenate(conf_outs, axis=1).squeeze(0)
        box = np.concatenate(box_outs, axis=1).squeeze(0)

        n = min(grids.shape[0], box.shape[0])
        grids, strides_arr, conf, box = grids[:n], strides_arr[:n], conf[:n], box[:n]

        x1 = (grids[:, 0] - box[:, 0]) * strides_arr[:, 0]
        y1 = (grids[:, 1] - box[:, 1]) * strides_arr[:, 0]
        x2 = (grids[:, 0] + box[:, 2]) * strides_arr[:, 0]
        y2 = (grids[:, 1] + box[:, 3]) * strides_arr[:, 0]
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        scores = conf[:, 0]

        mask = scores > self.conf_thres
        boxes, scores = boxes[mask], scores[mask]
        if len(boxes) == 0:
            return []

        boxes = unpad_and_scale(boxes, input_hw, origin_hw, pad_resize=True)
        keep = nms_numpy(boxes, scores, self.iou_thres)
        boxes, scores = boxes[keep], scores[keep]

        result = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            result.append(
                [
                    round(float(x1)),
                    round(float(y1)),
                    round(float(x2 - x1)),
                    round(float(y2 - y1)),
                    float(scores[i]),
                ]
            )
        return result
