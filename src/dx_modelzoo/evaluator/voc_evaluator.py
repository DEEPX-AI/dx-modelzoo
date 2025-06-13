from typing import Tuple
from collections import deque


import numpy as np
import torch
from tqdm import tqdm

from dx_modelzoo.dataset import DatasetBase
from dx_modelzoo.evaluator import EvaluatorBase
from dx_modelzoo.session import SessionBase
from dx_modelzoo.utils.detection import calculate_iou

from loguru import logger
import time

class VOC2007DetectionEvaluator(EvaluatorBase):
    """VOC2007 Evaluator for Object Detection.

    Args:
        session: runtime session.
        dataset: COCO dataset.
    """

    def __init__(self, session: SessionBase, dataset: DatasetBase):
        super().__init__(session, dataset)

    def eval(self):
        loader = self.make_loader()
        total_len = len(loader)
        results = []
        img_count = 0
        
        total_inference_time = 0.0
        recent_inference_times = deque(maxlen=50)  # total
        
        pbar = tqdm(loader, total=total_len)
        for images, origin_shape in pbar:
            
            start_time = time.time() 
            output = self.session.run(images)
            inference_time = time.time() - start_time
            
            recent_inference_times.append(inference_time)  
            total_inference_time += inference_time
            
            batched_boxes, batched_scores, batched_classes = self.postprocessing(output, session=self.session)
            for boxes, scores, classes in zip(batched_boxes, batched_scores, batched_classes):
                scaled_boxes = self._scale_boxes(boxes, origin_shape)
                data_id = torch.zeros_like(classes, dtype=torch.float32) + img_count
                result = torch.cat(
                    [
                        data_id.reshape(-1, 1),
                        classes.reshape(-1, 1).float(),
                        scores.reshape(-1, 1),
                        scaled_boxes,
                    ],
                    dim=1,
                )
                results.append(result)
                img_count += 1
            
            if len(recent_inference_times) > 0:
                current_fps = len(recent_inference_times) / sum(recent_inference_times)
            else:
                current_fps = 0.0

            pbar.desc = (
                f"VOC | Cuurent_FPS:{current_fps:.1f}"
            )
        results = torch.cat(results)
        avg_fps = total_len / total_inference_time if total_inference_time > 0 else 0.0

        map50 = self.calculate_mAP50(results)
        print(f"mAP@0.5: {round(map50*100, 3)}")
        print(f"Average FPS: {avg_fps:.2f}")
        logger.success(f"@JSON <mAP@0.5:{round(map50*100, 3)}; Average FPS:{avg_fps:.2f}>")
        
    def _scale_boxes(self, boxes: torch.Tensor, origin_shape: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        height, width, _ = origin_shape
        sclaed_boxes = boxes.clone()
        sclaed_boxes[..., 0] *= width
        sclaed_boxes[..., 1] *= height
        sclaed_boxes[..., 2] *= width
        sclaed_boxes[..., 3] *= height
        return sclaed_boxes

    def calculate_mAP50(self, results: torch.Tensor):
        """evaluate accuracy with mAP@0.5 type

        Args:
            results (torch.Tensor), (M, 7): inference results
                results[n, :] == [img_index, label, prob, x1, y1, x2, y2]
        """
        true_case_stat, all_gb_boxes, all_difficult_cases = self.dataset.group_annotation

        # {class_index: [] for class_index in range(1, len(self.class_names))}
        image_ids_dict = {}
        boxes_dict = {}
        probs_dict = {}
        for class_index, _ in enumerate(self.dataset.class_names):
            image_ids = []
            boxes = []
            probs = []
            if class_index == 0:
                continue  # ignore background
            sub = results[results[:, 1] == class_index, :]
            for i in range(sub.size(0)):
                prob_box = sub[i, 2:].numpy()
                image_id = self.dataset.data_ids[int(sub[i, 0])]
                image_ids.append(image_id)
                probs.append(float(prob_box[0]))

                box = torch.tensor([float(v) for v in prob_box[1:]]).unsqueeze(0)
                boxes.append(box)
            image_ids_dict[class_index] = image_ids
            boxes_dict[class_index] = boxes
            probs_dict[class_index] = probs

        aps = []
        print("\n\nAverage Precision Per-class:")
        logger.info("\n\nAverage Precision Per-class:")
        for class_index, class_name in enumerate(self.dataset.class_names):
            if class_index == 0:
                continue  # ignore background
            probs = np.array(probs_dict[class_index])
            sorted_indexes = np.argsort(-probs)
            sorted_boxes = [boxes_dict[class_index][i] for i in sorted_indexes]
            sorted_image_ids = [image_ids_dict[class_index][i] for i in sorted_indexes]
            ap = compute_average_precision_per_class(
                sorted_boxes,
                sorted_image_ids,
                true_case_stat[class_index],
                all_gb_boxes[class_index],
                all_difficult_cases[class_index],
                iou_threshold=0.5,
            )
            aps.append(ap)
            print(f"{class_name}: AP: {round(ap,5)}")
            logger.info(f"{class_name}: AP: {round(ap,5)}")
        return sum(aps) / len(aps)


def compute_average_precision_per_class(
    sorted_boxes,
    sorted_image_ids,
    num_true_cases,
    gt_boxes,
    difficult_cases,
    iou_threshold=0.5,
):
    true_positive = np.zeros(len(sorted_image_ids))
    false_positive = np.zeros(len(sorted_image_ids))
    matched = set()
    for i, image_id in enumerate(sorted_image_ids):
        box = sorted_boxes[i]
        if image_id not in gt_boxes:
            false_positive[i] = 1
            continue

        gt_box = gt_boxes[image_id]
        ious = calculate_iou(box, gt_box)
        max_iou = torch.max(ious).item()
        max_arg = torch.argmax(ious).item()
        if max_iou > iou_threshold:
            if difficult_cases[image_id][max_arg] == 0:
                if (image_id, max_arg) not in matched:
                    true_positive[i] = 1
                    matched.add((image_id, max_arg))
                else:
                    false_positive[i] = 1
        else:
            false_positive[i] = 1

    true_positive = true_positive.cumsum()
    false_positive = false_positive.cumsum()
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / num_true_cases
    return compute_voc2007_average_precision(precision, recall)


def compute_voc2007_average_precision(precision, recall):
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap = ap + p / 11.0
    return ap
