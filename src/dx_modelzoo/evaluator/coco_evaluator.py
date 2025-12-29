import json
import tempfile
import time
from collections import deque
from typing import List, Tuple

import torch
from loguru import logger
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from dx_modelzoo.dataset.coco import COCODataset
from dx_modelzoo.enums import SessionType
from dx_modelzoo.evaluator import EvaluatorBase
from dx_modelzoo.evaluator.constant import COCO80TO91MAPPER
from dx_modelzoo.session import SessionBase
from dx_modelzoo.utils.detection import (
    convert_xyxy_2_cxcywh,
    get_pad_size,
    get_ratios,
    scale_boxes,
)

COCO_DET = List[int | torch.Tensor]


class COCOEvaluator(EvaluatorBase):
    """COCO Evaluator for obeject detection

    Args:
        session (SessionBase): runtime session.
        dataset (COCODataset): COCO dataset.
    """

    def __init__(self, session: SessionBase, dataset: COCODataset):
        super().__init__(session, dataset)
        self.dataset: COCODataset
        self.total_inference_time = 0.0  # Total time spent on inference
        self.recent_inference_times = deque(maxlen=50)  # total: 5000

    def eval(self) -> None:
        """evaluation OD model with COCO dataset."""
        loader = self.make_loader()
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

                if len(self.recent_inference_times) > 0:
                    current_fps = len(self.recent_inference_times) / sum(self.recent_inference_times)
                else:
                    current_fps = 0.0

                pbar.desc = f"COCO | Current_FPS:{current_fps:.1f}"
            temp_f.write("\n]\n")
            temp_f.flush()
            mAP, mAP50 = self._run_coco_eval(temp_f.name, self.dataset.coco_annotation)

        avg_fps = total_len / self.total_inference_time if self.total_inference_time > 0 else 0

        print(f"mAP: {round(mAP*100, 3)} mAP50: {round(mAP50*100, 3)}")
        print(f"Average Inference FPS: {avg_fps:.2f}")
        logger.success(f"@JSON <mAP:{round(mAP*100, 3)}; mAP50:{round(mAP50*100, 3)}; Average FPS:{avg_fps:.2f}>")

        return {
            "performance": [mAP*100, mAP50*100], 
            "fps": avg_fps,
        }

    def _run_one_batch(self, batch: Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]) -> str:
        """run one batch.

        model output boxes format is cxcywh format.
        run_one_batch output boxes format is same with model output boxes format.

        Args:
            batch (Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]): one batch.
        """
        image, origin_shape, img_id = batch
        origin_shape = [value[0] for value in origin_shape]

        start_time = time.time()
        outputs = self.session.run(image)
        inference_time = time.time() - start_time

        self.recent_inference_times.append(inference_time)
        self.total_inference_time += inference_time

        outputs = self.postprocessing(outputs)
        if self.session.type == SessionType.dxruntime:
            image = image.permute(0, 3, 1, 2)
        scaled_boxes = self._change_box_scales_to_origin(image, origin_shape, outputs)
        scaled_boxes = convert_xyxy_2_cxcywh(scaled_boxes)
        coco_formatted_dets = self._make_coco_format_det(scaled_boxes, outputs, img_id)

        if not coco_formatted_dets:
            return ""

        return ",\n".join([json.dumps(det) for det in coco_formatted_dets])

    def _change_box_scales_to_origin(
        self, image: torch.Tensor, origin_shape: List[torch.Tensor], outputs: torch.Tensor
    ) -> torch.Tensor:
        """change output bounding boxes scales to origin image.
        outputs's boxes format is xyxy format.

        Args:
            image (torch.Tensor): origin image.
            origin_shape (List[torch.Tensor]): origin image shape.
            outputs (np.ndarray): model outputs.

        Returns:
            np.ndarray: scaled boxes.
        """
        cloned = outputs.clone()
        ratios = get_ratios(image, origin_shape)
        pads = get_pad_size(image, origin_shape, ratios)
        return scale_boxes(cloned[..., :4], origin_shape, ratios, pads)

    def _make_coco_format_det(self, boxes: torch.Tensor, outputs: torch.Tensor, img_id: torch.Tensor) -> List[COCO_DET]:
        """make coco det.
        boxes format is cxcywh format.

        Args:
            boxes (np.ndarray): scaled boxes.
            outputs (np.ndarray): model outputs.
            img_id (torch.Tensor): coco image id.

        Returns:
            List[COCO_DET]: coco det list.
        """
        detections = []
        boxes_list = boxes.cpu().numpy().tolist()
        outputs_list = outputs.cpu().numpy().tolist()
        image_id = int(img_id.item())

        for box, output in zip(boxes_list, outputs_list):
            detection = {
                "image_id": image_id,
                "category_id": COCO80TO91MAPPER[int(output[5])],
                "bbox": [round(coord, 3) for coord in box],
                "score": round(output[4], 5),
            }
            detections.append(detection)
        return detections

    def _run_coco_eval(self, result_file_path: str, coco_annotation: COCO) -> None:
        """run coco evaluation.

        Args:
            coco_det_list (List[List]): coco det list.
            coco_annotation (COCO): coco annoation.
        """
        
        predicted_det = coco_annotation.loadRes(result_file_path)

        coco_eval = COCOeval(coco_annotation, predicted_det, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        mAP, mAP50 = coco_eval.stats[:2]
        return mAP, mAP50
