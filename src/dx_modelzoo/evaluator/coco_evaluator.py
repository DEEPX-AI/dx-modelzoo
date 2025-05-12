from typing import List, Tuple
from collections import deque

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from dx_modelzoo.dataset.coco import COCODataset
from dx_modelzoo.evaluator import EvaluatorBase
from dx_modelzoo.evaluator.constant import COCO80TO91MAPPER
from dx_modelzoo.session import SessionBase
from dx_modelzoo.utils.detection import convert_xyxy_2_cxcywh, get_pad_size, get_ratios, scale_boxes

from loguru import logger
import time

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
        self.recent_inference_times = deque(maxlen=50) # total: 5000

    def eval(self) -> None:
        """evaluation OD model with COCO dataset."""
        loader = self.make_loader()
        total_len = len(loader)

        pbar = tqdm(loader, total=total_len)
        coco_det_list = []
        for batch in pbar:
            coco_det_list.extend(self._run_one_batch(batch))
            
            if len(self.recent_inference_times) > 0:
                current_fps = len(self.recent_inference_times) / sum(self.recent_inference_times)
            else:
                current_fps = 0.0
                
            pbar.desc = (
                f"COCO | Current_FPS:{current_fps:.1f}"
            )
        mAP, mAP50 = self._run_coco_eval(coco_det_list, self.dataset.coco_annotation)
        avg_fps = total_len / self.total_inference_time if self.total_inference_time > 0 else 0
        
        print(f"mAP: {round(mAP*100, 3)} mAP50: {round(mAP50*100, 3)}")
        print(f"Average Inference FPS: {avg_fps:.2f}")       
        logger.success(f"@JSON <mAP:{round(mAP*100, 3)}; mAP50:{round(mAP50*100, 3)}; Average FPS:{avg_fps:.2f}>")
        

    def _run_one_batch(self, batch: Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]) -> List[COCO_DET]:
        """run one batch.

        model output boxes format is cxcywh format.
        run_one_batch output boxes format is same with model output boxes format.

        Args:
            batch (Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]): one batch.

        Returns:
            List[COCO_DET]: COCO dets format list.
        """
        image, origin_shape, img_id = batch
        origin_shape = [value[0] for value in origin_shape]

        start_time = time.time()
        outputs = self.session.run(image)
        inference_time = time.time() - start_time
        
        self.recent_inference_times.append(inference_time) 
        self.total_inference_time += inference_time
         
        outputs = self.postprocessing(outputs)
        scaled_boxes = self._change_box_scales_to_origin(image, origin_shape, outputs)
        scaled_boxes = convert_xyxy_2_cxcywh(scaled_boxes)
        return self._make_coco_format_det(scaled_boxes, outputs, img_id)

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
        return [[int(img_id), *box, output[4], COCO80TO91MAPPER[int(output[5])]] for box, output in zip(boxes, outputs)]

    def _run_coco_eval(self, coco_det_list: List[List], coco_anntoation: COCO) -> None:
        """run coco evaluation.

        Args:
            coco_det_list (List[List]): coco det list.
            coco_anntoation (COCO): coco annoation.
        """
        coco_det_list = sorted(coco_det_list, key=lambda x: x[0])
        coco_det_list = np.array(coco_det_list, dtype=np.float32)
        predicted_det = coco_anntoation.loadRes(coco_det_list)

        coco_eval = COCOeval(coco_anntoation, predicted_det, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        mAP, mAP50 = coco_eval.stats[:2]
        return mAP, mAP50
        
