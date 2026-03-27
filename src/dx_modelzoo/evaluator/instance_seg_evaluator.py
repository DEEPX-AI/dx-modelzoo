import json
import tempfile
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools.mask import encode as rle_encode

from dx_modelzoo.dataset import DatasetBase
from dx_modelzoo.evaluator import EvaluatorBase
from dx_modelzoo.evaluator.constant import COCO80TO91MAPPER
from dx_modelzoo.session import SessionBase


def crop_mask(masks: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    n, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]
    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def process_masks(
    protos: torch.Tensor, masks_in: torch.Tensor, bboxes: torch.Tensor, input_shape: tuple, original_shape: tuple
) -> np.ndarray:
    c, mh, mw = protos.shape
    ih, iw = input_shape
    h_orig, w_orig = original_shape

    masks = (masks_in @ protos.float().view(c, -1)).sigmoid().view(-1, mh, mw)
    masks = F.interpolate(masks[None], size=(ih, iw), mode="bilinear", align_corners=False)[0]
    masks = crop_mask(masks, bboxes)

    gain = min(ih / h_orig, iw / w_orig)
    pad_x = (iw - w_orig * gain) / 2
    pad_y = (ih - h_orig * gain) / 2
    top, left = int(round(pad_y)), int(round(pad_x))
    bottom, right = int(round(ih - pad_y)), int(round(iw - pad_x))

    masks = masks[:, top:bottom, left:right]
    masks = F.interpolate(masks[None], size=(h_orig, w_orig), mode="bilinear", align_corners=False)[0]

    return masks.gt_(0.5).cpu().numpy().astype(np.uint8)


class InstanceSegEvaluator(EvaluatorBase):
    """COCO Evaluator for instance segmentation"""

    def __init__(self, session: SessionBase, dataset: DatasetBase):
        super().__init__(session, dataset, workers=12)
        self.coco80_to_91_map = COCO80TO91MAPPER
        self.temp_file = None
        self.first_chunk_written = False
        self.lazy_postprocessing = True

    def init_metrics(self) -> dict:
        """Initialize temporary file for segmentation results."""
        self.temp_file = tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8", suffix=".json", delete=False)
        self.temp_file.write("[\n")
        self.first_chunk_written = False
        return {"detections_count": 0}

    def extract_inputs(self, batch_data) -> torch.Tensor:
        """Extract image from batch."""
        image, origin_shape_tensors, img_id_tensor = batch_data
        return image

    def process_batch_result(self, batch_data, output, metrics_state: dict) -> dict:
        """Process batch result and write segmentation masks to file."""
        image, origin_shape_tensors, img_id_tensor = batch_data
        origin_shape = (origin_shape_tensors[0].item(), origin_shape_tensors[1].item())
        img_id = img_id_tensor.item()

        if self.lazy_postprocessing:
            predictions, prototypes = output

            # Workaround
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
            final_masks = process_masks(
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
        """Compute final segmentation metrics."""
        # Finalize JSON file
        self.temp_file.write("\n]\n")
        self.temp_file.flush()
        self.temp_file.close()

        # Run COCO evaluation
        print("\n--- Segmentation Evaluation ---")
        mAP, mAP50 = self._run_coco_eval(self.temp_file.name, self.dataset.coco_annotation)

        # Clean up
        import os

        os.unlink(self.temp_file.name)

        # Calculate FPS
        avg_fps = (
            metrics_state["detections_count"] / self.total_inference_time if self.total_inference_time > 0 else 0.0
        )

        print("\n" + "=" * 30)
        print("        Final Segmentation Results")
        print("=" * 30)
        print(f"mAP: {round(mAP*100, 3)} mAP50: {round(mAP50*100, 3)}")
        print(f"Average Inference FPS: {avg_fps:.2f}")
        logger.success(f"@JSON <mAP:{round(mAP*100, 3)}; mAP50:{round(mAP50*100, 3)}; Average FPS:{avg_fps:.2f}>")

        return {
            "performance": [mAP * 100, mAP50 * 100],
            "fps": avg_fps,
        }

    def format_progress_desc(self, metrics_state: dict, current_fps: float) -> str:
        """Format progress bar."""
        det_count = metrics_state.get("detections_count", 0)
        return f"COCO | Dets:{det_count} Current_FPS:{current_fps:.1f}"

    def _make_coco_format_seg(self, masks: np.ndarray, outputs: torch.Tensor, img_id: int) -> List[dict]:
        """Make COCO segmentation result format."""
        seg_detections = []
        outputs_np = outputs.cpu().numpy()

        for i in range(len(outputs_np)):
            output = outputs_np[i]
            category_id = (
                int(output[5]) if getattr(self, "use_class_90", False) else self.coco80_to_91_map[int(output[5])]
            )
            score = round(float(output[4]), 5)

            rle = rle_encode(np.asfortranarray(masks[i]))
            rle["counts"] = rle["counts"].decode("utf-8")

            seg_det = {
                "image_id": img_id,
                "category_id": category_id,
                "segmentation": rle,
                "score": score,
            }
            seg_detections.append(seg_det)

        return seg_detections

    def _run_coco_eval(self, result_file_path: str, coco_annotation: COCO) -> Tuple[float, float]:
        """Run COCO evaluation for segmentation."""
        try:
            predicted_det = coco_annotation.loadRes(result_file_path)
            coco_eval = COCOeval(coco_annotation, predicted_det, "segm")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            mAP, mAP50 = coco_eval.stats[:2]
            return mAP, mAP50
        except Exception as e:
            logger.error(f"Error during COCO segmentation evaluation: {e}")
            return 0.0, 0.0
