import json
import tempfile
import time
from collections import deque
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools.mask import encode as rle_encode
from tqdm import tqdm

from dx_modelzoo.evaluator import EvaluatorBase
from dx_modelzoo.evaluator.constant import COCO80TO91MAPPER


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

    def __init__(self, session, dataset):
        super().__init__(session, dataset)
        self.total_inference_time = 0.0
        self.recent_inference_times = deque(maxlen=50)
        self.coco80_to_91_map = COCO80TO91MAPPER

    def eval(self) -> None:
        """Evaluation instance segmentation model with COCO dataset."""
        loader = self.make_loader()  # Assuming dataset is iterable
        total_len = len(loader)

        pbar = tqdm(loader, total=total_len)
        with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8", suffix=".json") as temp_f:
            temp_f.write("[\n")

            first_chunk_written = False
            for batch in pbar:
                json_chunk = self._run_one_batch(batch)
                if json_chunk:
                    if first_chunk_written:
                        temp_f.write(",\n")
                    temp_f.write(json_chunk)
                    first_chunk_written = True

                current_fps = (
                    len(self.recent_inference_times) / sum(self.recent_inference_times)
                    if self.recent_inference_times
                    else 0.0
                )
                pbar.desc = f"COCO | Current_FPS:{current_fps:.1f}"

            temp_f.write("\n]\n")
            temp_f.flush()

            print("\n--- Segmentation Evaluation ---")
            mAP, mAP50 = self._run_coco_eval(temp_f.name, self.dataset.coco_annotation)

        avg_fps = total_len / self.total_inference_time if self.total_inference_time > 0 else 0

        print("\n" + "=" * 30)
        print("        Final Segmentation Results")
        print("=" * 30)
        print(f"mAP: {round(mAP*100, 3)} mAP50: {round(mAP50*100, 3)}")
        print(f"Average Inference FPS: {avg_fps:.2f}")
        logger.success(f"@JSON <mAP:{round(mAP*100, 3)}; mAP50:{round(mAP50*100, 3)}; Average FPS:{avg_fps:.2f}>")

        return {
            "performance": [mAP*100, mAP50*100], 
            "fps": avg_fps,
        }

    def _run_one_batch(self, batch) -> str:
        """Run one batch for instance segmentation."""
        image, origin_shape_tensors, img_id_tensor = batch
        origin_shape = (origin_shape_tensors[0].item(), origin_shape_tensors[1].item())
        img_id = img_id_tensor.item()

        start_time = time.time()
        predictions, prototypes = self.session.run(image)

        # Workaround
        if predictions.shape[1] == 32:
            predictions, prototypes = prototypes, predictions
        inference_time = time.time() - start_time

        self.recent_inference_times.append(inference_time)
        self.total_inference_time += inference_time

        predictions = torch.from_numpy(predictions)
        prototypes = torch.from_numpy(prototypes)

        outputs = self.postprocessing(predictions)[0]
        proto = prototypes[0]

        if outputs is None or len(outputs) == 0:
            return ""

        final_masks = process_masks(
            proto,
            outputs[:, 6:],
            outputs[:, :4],
            input_shape=[640, 640],
            original_shape=origin_shape,
        )

        coco_formatted_dets = self._make_coco_format_seg(final_masks, outputs, img_id)

        if not coco_formatted_dets:
            return ""

        return ",\n".join([json.dumps(det) for det in coco_formatted_dets])

    def _make_coco_format_seg(self, masks: np.ndarray, outputs: torch.Tensor, img_id: int) -> List[dict]:
        """Make COCO segmentation result format."""
        seg_detections = []
        outputs_list = outputs.cpu().numpy().tolist()

        for i, output in enumerate(outputs_list):
            category_id = self.coco80_to_91_map[int(output[5])]
            score = round(output[4], 5)

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
