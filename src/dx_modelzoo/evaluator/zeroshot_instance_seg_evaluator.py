"""Zero-shot instance segmentation evaluator.

Evaluates class-agnostic instance segmentation (e.g., FastSAM)
using COCO metrics with useCats=0 for class-agnostic AR/AP.
"""

import json
import tempfile
from typing import List, Tuple

import cv2
import numpy as np
import torch
from loguru import logger
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools.mask import encode as rle_encode

from dx_modelzoo.dataset import DatasetBase
from dx_modelzoo.evaluator import EvaluatorBase
from dx_modelzoo.evaluator.instance_seg_evaluator import crop_mask
from dx_modelzoo.session import SessionBase


def process_masks_fast(
    protos: torch.Tensor, masks_in: torch.Tensor, bboxes: torch.Tensor, input_shape: tuple, original_shape: tuple
) -> np.ndarray:
    """Process masks at prototype resolution and upscale binary results with nearest interpolation."""
    c, mh, mw = protos.shape
    ih, iw = input_shape
    h_orig, w_orig = original_shape

    masks = (masks_in @ protos.float().view(c, -1)).sigmoid().view(-1, mh, mw)

    scale_x, scale_y = mw / iw, mh / ih
    scaled_boxes = bboxes.clone()
    scaled_boxes[:, [0, 2]] *= scale_x
    scaled_boxes[:, [1, 3]] *= scale_y
    masks = crop_mask(masks, scaled_boxes)
    masks = masks.gt_(0.5).cpu().numpy().astype(np.uint8)

    gain = min(ih / h_orig, iw / w_orig)
    pad_x, pad_y = (iw - w_orig * gain) / 2, (ih - h_orig * gain) / 2
    top = int(round(pad_y * scale_y))
    left = int(round(pad_x * scale_x))
    bottom = int(round((ih - pad_y) * scale_y))
    right = int(round((iw - pad_x) * scale_x))
    masks = masks[:, top:bottom, left:right]

    result = np.empty((len(masks), h_orig, w_orig), dtype=np.uint8)
    for i in range(len(masks)):
        result[i] = cv2.resize(masks[i], (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    return result


class ZeroShotInstanceSegEvaluator(EvaluatorBase):
    """COCO Evaluator for zero-shot (class-agnostic) instance segmentation.

    All predictions are mapped to category_id=1 and evaluated with useCats=0.
    """

    def __init__(self, session: SessionBase, dataset: DatasetBase):
        super().__init__(session, dataset, workers=12)
        self.temp_file = None
        self.first_chunk_written = False
        self.lazy_postprocessing = True

    def init_metrics(self) -> dict:
        self.temp_file = tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8", suffix=".json", delete=False)
        self.temp_file.write("[\n")
        self.first_chunk_written = False
        return {"detections_count": 0}

    def extract_inputs(self, batch_data) -> torch.Tensor:
        image, origin_shape_tensors, img_id_tensor = batch_data
        return image

    def process_batch_result(self, batch_data, output, metrics_state: dict) -> dict:
        image, origin_shape_tensors, img_id_tensor = batch_data
        origin_shape = (origin_shape_tensors[0].item(), origin_shape_tensors[1].item())
        img_id = img_id_tensor.item()

        if self.lazy_postprocessing:
            predictions, prototypes = output

            if predictions.shape[1] == 32:
                predictions, prototypes = prototypes, predictions

            predictions = torch.from_numpy(predictions)
            prototypes = torch.from_numpy(prototypes)

            outputs = self.postprocessing(predictions)[0]
            proto = prototypes[0]
        else:
            outputs = output[0]
            proto = output[1]

        if outputs is not None and len(outputs) > 0:
            final_masks = process_masks_fast(
                proto,
                outputs[:, 6:],
                outputs[:, :4],
                input_shape=image.shape[2:] if len(image.shape) == 4 and image.shape[1] == 3 else image.shape[1:3],
                original_shape=origin_shape,
            )

            coco_formatted_dets = self._make_coco_format_seg(final_masks, outputs, img_id)

            if coco_formatted_dets:
                json_chunk = ",\n".join([json.dumps(det) for det in coco_formatted_dets])
                if self.first_chunk_written:
                    self.temp_file.write(",\n")
                self.temp_file.write(json_chunk)
                self.first_chunk_written = True
                metrics_state["detections_count"] += len(coco_formatted_dets)

        return metrics_state

    def compute_final_metrics(self, metrics_state: dict) -> dict:
        self.temp_file.write("\n]\n")
        self.temp_file.flush()
        self.temp_file.close()

        print("\n--- Zero-Shot Instance Segmentation Evaluation ---")
        ar10, ar100, ar1000 = self._run_coco_eval(self.temp_file.name, self.dataset.coco_annotation)

        import os

        os.unlink(self.temp_file.name)

        avg_fps = (
            metrics_state["detections_count"] / self.total_inference_time if self.total_inference_time > 0 else 0.0
        )

        print(f"\nAR@10: {round(ar10*100, 3)} AR@100: {round(ar100*100, 3)} AR@1000: {round(ar1000*100, 3)}")
        print(f"Average Inference FPS: {avg_fps:.2f}")
        logger.success(
            f"@JSON <AR@10:{round(ar10*100, 3)}; AR@100:{round(ar100*100, 3)}; "
            f"AR@1000:{round(ar1000*100, 3)}; Average FPS:{avg_fps:.2f}>"
        )

        return {
            "performance": [ar10 * 100, ar100 * 100, ar1000 * 100],
            "fps": avg_fps,
        }

    def format_progress_desc(self, metrics_state: dict, current_fps: float) -> str:
        det_count = metrics_state.get("detections_count", 0)
        return f"COCO | Dets:{det_count} Current_FPS:{current_fps:.1f}"

    def _make_coco_format_seg(self, masks: np.ndarray, outputs: torch.Tensor, img_id: int) -> List[dict]:
        seg_detections = []
        outputs_np = outputs.cpu().numpy()

        for i in range(len(outputs_np)):
            output = outputs_np[i]
            score = round(float(output[4]), 5)

            rle = rle_encode(np.asfortranarray(masks[i]))
            rle["counts"] = rle["counts"].decode("utf-8")

            seg_det = {
                "image_id": img_id,
                "category_id": 1,
                "segmentation": rle,
                "score": score,
            }
            seg_detections.append(seg_det)

        return seg_detections

    def _run_coco_eval(self, result_file_path: str, coco_annotation: COCO) -> Tuple[float, float, float]:
        try:
            predicted_det = coco_annotation.loadRes(result_file_path)
            coco_eval = COCOeval(coco_annotation, predicted_det, "segm")
            coco_eval.params.useCats = 0
            coco_eval.params.maxDets = [10, 100, 1000]
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            # stats[6] = AR@maxDets[0]=10, stats[7] = AR@maxDets[1]=100, stats[8] = AR@maxDets[2]=1000
            ar10 = coco_eval.stats[6]
            ar100 = coco_eval.stats[7]
            ar1000 = coco_eval.stats[8]
            return ar10, ar100, ar1000
        except Exception as e:
            logger.error(f"Error during COCO evaluation: {e}")
            return 0.0, 0.0, 0.0
