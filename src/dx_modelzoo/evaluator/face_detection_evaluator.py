from __future__ import annotations

import os
from collections import defaultdict
from typing import Any, Dict, Tuple

import numpy as np
from tqdm import tqdm

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.evaluator import EVALUATOR_REGISTRY, EvaluatorBase
from dx_modelzoo.session import SessionBase


@EVALUATOR_REGISTRY.register("face_detection")
class FaceDetectionEvaluator(EvaluatorBase):
    """WiderFace Evaluator for Face Detection (AP)."""

    def __init__(self, session: SessionBase, dataset: DatasetBase, **kwargs) -> None:
        super().__init__(session, dataset, workers=12, **kwargs)
        self._predictions: Dict[str, Dict[str, np.ndarray]] = defaultdict(dict)

    def init_metrics(self) -> dict:
        self._predictions = defaultdict(dict)
        return {}

    def extract_inputs(self, batch_data: Tuple) -> np.ndarray:
        image, origin_shape, path = batch_data
        return image

    def _build_postprocessing_context(self, batch_data) -> dict:
        image, origin_shape, _path = batch_data
        if isinstance(origin_shape, (list, tuple)):
            origin_shape = [int(v[0]) if hasattr(v, "__getitem__") else int(v) for v in origin_shape]
        origin_hw = (int(origin_shape[0]), int(origin_shape[1]))
        if image.ndim >= 3 and image.shape[-1] in (1, 3):
            input_hw = (image.shape[-3], image.shape[-2])
        else:
            input_hw = (image.shape[-2], image.shape[-1])
        return {"origin_hw": origin_hw, "input_hw": input_hw}

    def process_batch_result(self, batch_data: Tuple, output: Any, metrics_state: dict) -> dict:
        image, origin_shape, path = batch_data
        path = path[0] if isinstance(path, (list, tuple)) else path

        # output is already postprocessed by _run_postprocessing
        outputs = output
        boxes = (
            np.array([[float(out[i]) for i in range(5)] for out in outputs]) if len(outputs) > 0 else np.empty((0, 5))
        )

        # Store in-memory: group by event_folder / image_name
        pict_folder = path.split(os.path.sep)[-2]
        image_name = os.path.splitext(os.path.basename(path))[0]
        self._predictions[pict_folder][image_name] = boxes
        return metrics_state

    def compute_final_metrics(self, metrics_state: dict) -> dict:
        total_len = len(self.dataset)
        avg_fps = total_len / self.total_inference_time if self.total_inference_time > 0 else 0.0
        aps = self.evaluation(self._predictions)
        return self._finalize(
            metric_names=["Easy AP", "Medium AP", "Hard AP"],
            metric_values=[aps[0] * 100, aps[1] * 100, aps[2] * 100],
            fps=avg_fps,
        )

    def format_progress_desc(self, metrics_state: dict, current_fps: float) -> str:
        return f"WiderFace | Current_FPS:{current_fps:.1f}"

    def evaluation(self, pred, iou_thresh=0.5):
        self.norm_score(pred)
        (facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list) = self.dataset.get_gt_boxes()
        event_num = len(event_list)
        thresh_num = 1000
        settings = ["easy", "medium", "hard"]
        setting_gts = [easy_gt_list, medium_gt_list, hard_gt_list]
        aps = []
        for setting_id in range(3):
            gt_list = setting_gts[setting_id]
            count_face = 0
            pr_curve = np.zeros((thresh_num, 2)).astype("float")
            pbar = tqdm(range(event_num))
            for i in pbar:
                pbar.set_description("Processing {}".format(settings[setting_id]))
                event_name = str(event_list[i][0][0])
                img_list = file_list[i][0]
                pred_list = pred[event_name]
                sub_gt_list = gt_list[i][0]
                gt_bbx_list = facebox_list[i][0]
                for j in range(len(img_list)):
                    pred_info = pred_list[str(img_list[j][0][0])]
                    gt_boxes = gt_bbx_list[j][0].astype("float")
                    keep_index = sub_gt_list[j][0]
                    count_face += len(keep_index)
                    if len(gt_boxes) == 0 or len(pred_info) == 0:
                        continue
                    ignore = np.zeros(gt_boxes.shape[0])
                    if len(keep_index) != 0:
                        ignore[keep_index - 1] = 1
                    pred_recall, proposal_list = self.image_eval(pred_info, gt_boxes, ignore, iou_thresh)
                    _img_pr_info = self.img_pr_info(thresh_num, pred_info, proposal_list, pred_recall)
                    pr_curve += _img_pr_info
            pr_curve = self.dataset_pr_info(thresh_num, pr_curve, count_face)
            ap = self.voc_ap(pr_curve[:, 1], pr_curve[:, 0])
            aps.append(ap)
        return aps

    def voc_ap(self, rec, prec):
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        i = np.where(mrec[1:] != mrec[:-1])[0]
        return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))

    def dataset_pr_info(self, thresh_num, pr_curve, count_face):
        _pr_curve = np.zeros((thresh_num, 2))
        for i in range(thresh_num):
            _pr_curve[i, 0] = pr_curve[i, 1] / pr_curve[i, 0] if pr_curve[i, 0] > 0 else 0
            _pr_curve[i, 1] = pr_curve[i, 1] / count_face if count_face > 0 else 0
        return _pr_curve

    def img_pr_info(self, thresh_num, pred_info, proposal_list, pred_recall):
        pr_info = np.zeros((thresh_num, 2)).astype("float")
        for t in range(thresh_num):
            thresh = 1 - (t + 1) / thresh_num
            r_index = np.where(pred_info[:, 4] >= thresh)[0]
            if len(r_index) == 0:
                pr_info[t, 0] = 0
                pr_info[t, 1] = 0
            else:
                r_index = r_index[-1]
                p_index = np.where(proposal_list[: r_index + 1] == 1)[0]
                pr_info[t, 0] = len(p_index)
                pr_info[t, 1] = pred_recall[r_index]
        return pr_info

    def image_eval(self, pred, gt, ignore, iou_thresh):
        _pred = pred.copy()
        _gt = gt.copy()
        pred_recall = np.zeros(_pred.shape[0])
        recall_list = np.zeros(_gt.shape[0])
        proposal_list = np.ones(_pred.shape[0])
        _pred[:, 2] = _pred[:, 2] + _pred[:, 0]
        _pred[:, 3] = _pred[:, 3] + _pred[:, 1]
        _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
        _gt[:, 3] = _gt[:, 3] + _gt[:, 1]
        overlaps = self.bbox_overlaps(_pred[:, :4], _gt)
        for h in range(_pred.shape[0]):
            gt_overlap = overlaps[h]
            max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()
            if max_overlap >= iou_thresh:
                if ignore[max_idx] == 0:
                    recall_list[max_idx] = -1
                    proposal_list[h] = -1
                elif recall_list[max_idx] == 0:
                    recall_list[max_idx] = 1
            r_keep_index = np.where(recall_list == 1)[0]
            pred_recall[h] = len(r_keep_index)
        return pred_recall, proposal_list

    def bbox_overlaps(self, boxes, query_boxes):
        N = boxes.shape[0]
        K = query_boxes.shape[0]
        overlaps = np.zeros((N, K), dtype=np.float64)
        for k in range(K):
            box_area = (query_boxes[k, 2] - query_boxes[k, 0] + 1) * (query_boxes[k, 3] - query_boxes[k, 1] + 1)
            for n in range(N):
                iw = min(boxes[n, 2], query_boxes[k, 2]) - max(boxes[n, 0], query_boxes[k, 0]) + 1
                if iw > 0:
                    ih = min(boxes[n, 3], query_boxes[k, 3]) - max(boxes[n, 1], query_boxes[k, 1]) + 1
                    if ih > 0:
                        ua = float(
                            (boxes[n, 2] - boxes[n, 0] + 1) * (boxes[n, 3] - boxes[n, 1] + 1) + box_area - iw * ih
                        )
                        overlaps[n, k] = iw * ih / ua
        return overlaps

    def norm_score(self, pred):
        max_score = 0
        min_score = 1
        for _, k in pred.items():
            for _, v in k.items():
                if len(v) == 0:
                    continue
                _min = np.min(v[:, -1])
                _max = np.max(v[:, -1])
                max_score = max(_max, max_score)
                min_score = min(_min, min_score)
        diff = max_score - min_score
        for _, k in pred.items():
            for _, v in k.items():
                if len(v) == 0:
                    continue
                v[:, -1] = (v[:, -1] - min_score) / diff if diff > 0 else v[:, -1]
