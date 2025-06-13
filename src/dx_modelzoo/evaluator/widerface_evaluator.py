from collections import deque
import os
import shutil

import numpy as np
from tqdm import tqdm

from dx_modelzoo.evaluator import EvaluatorBase

from loguru import logger
import time

class WiderFaceEvaluator(EvaluatorBase):
    """WiderFace Evaluator for Face Detection.

    Args:
        session: runtime session.
        dataset: COCO dataset.
    """

    def __init__(self, session, dataset):
        super().__init__(session, dataset)

        self.save_folder = os.path.expanduser("~/.cache/wider_txt/")
        if not os.path.isdir(self.save_folder):
            os.makedirs(self.save_folder)

    def eval(self):
        loader = self.make_loader()
        total_len = len(loader)
        total_inference_time = 0.0
        recent_inference_times = deque(maxlen=30)  # total: 3226 
        
        pbar = tqdm(loader, total= total_len)
        for image, origin_shape, path in pbar:
            path = path[0]
            origin_shape = [int(value[0]) for value in origin_shape]
            h, w, _ = origin_shape
            start_time = time.time() 
            outputs = self.session.run(image)
            inference_time = time.time() - start_time
            
            recent_inference_times.append(inference_time)  
            total_inference_time += inference_time
            
            outputs = self.postprocessing(outputs, image.shape, origin_shape, self.session)
            boxes = []
            for output in outputs:
                box = [float(output[i]) for i in range(5)]
                boxes.append(box)
            pict_folder = path.split(os.path.sep)[-2]
            image_name = os.path.basename(path)
            txt_name = os.path.splitext(image_name)[0] + ".txt"
            save_name = os.path.join(self.save_folder, pict_folder, txt_name)
            dirname = os.path.dirname(save_name)
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
            with open(save_name, "w") as fd:
                file_name = os.path.basename(save_name)[:-4] + "\n"
                bboxs_num = str(len(boxes)) + "\n"
                fd.write(file_name)
                fd.write(bboxs_num)
                for box in boxes:
                    fd.write(
                        "%d %d %d %d %.03f" % (box[0], box[1], box[2], box[3], box[4] if box[4] <= 1 else 1) + "\n"
                    )
            if len(recent_inference_times) > 0:
                current_fps = len(recent_inference_times) / sum(recent_inference_times)
            else:
                current_fps = 0.0
                
            pbar.desc = (
                f"WiderFace | Current_FPS:{current_fps:.1f} "
            )
                            
        avg_fps = total_len / total_inference_time if total_inference_time > 0 else 0.0
        
        print("Finish saving results")
        aps = self.evaluation(pred=self.save_folder)
        print(f"Easy   Val AP: {round(aps[0] * 100, 3)}")
        print(f"Medium Val AP: {round(aps[1] * 100, 3)}")
        print(f"Hard   Val AP: {round(aps[2] * 100, 3)}")
        print(f"Average FPS: {avg_fps:.2f}") 
        logger.success(f"@JSON <Easy Val AP:{round(aps[0] * 100, 3)}; Medium Val AP:{round(aps[1] * 100, 3)}; Hard Val AP:{round(aps[2] * 100, 3)}; Average FPS:{avg_fps:.2f}>")
        self.remove_cache()
        return aps

    def evaluation(self, pred, iou_thresh=0.5):
        pred = self.get_preds(pred)
        self.norm_score(pred)
        (
            facebox_list,
            event_list,
            file_list,
            hard_gt_list,
            medium_gt_list,
            easy_gt_list,
        ) = self.dataset.get_gt_boxes()
        event_num = len(event_list)
        thresh_num = 1000
        settings = ["easy", "medium", "hard"]
        setting_gts = [easy_gt_list, medium_gt_list, hard_gt_list]
        aps = []
        for setting_id in range(3):
            # different setting
            gt_list = setting_gts[setting_id]
            count_face = 0
            pr_curve = np.zeros((thresh_num, 2)).astype("float")
            # [hard, medium, easy]

            pbar = tqdm(range(event_num))
            for i in pbar:
                pbar.set_description("Processing {}".format(settings[setting_id]))
                event_name = str(event_list[i][0][0])
                img_list = file_list[i][0]
                pred_list = pred[event_name]
                sub_gt_list = gt_list[i][0]
                # img_pr_info_list = np.zeros((len(img_list), thresh_num, 2))
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

            propose = pr_curve[:, 0]
            recall = pr_curve[:, 1]

            ap = self.voc_ap(recall, propose)
            aps.append(ap)
        return aps

    def voc_ap(self, rec, prec):
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    def dataset_pr_info(self, thresh_num, pr_curve, count_face):
        _pr_curve = np.zeros((thresh_num, 2))
        for i in range(thresh_num):
            _pr_curve[i, 0] = pr_curve[i, 1] / pr_curve[i, 0]
            _pr_curve[i, 1] = pr_curve[i, 1] / count_face
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
        """single image evaluation
        pred: Nx5
        gt: Nx4
        ignore:
        """
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
        """
        Parameters
        ----------
        boxes: (N, 4) ndarray of float
        query_boxes: (K, 4) ndarray of float
        Returns
        -------
        overlaps: (N, K) ndarray of overlap between boxes and query_boxes
        """
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
        """norm score
        pred {key: [[x1,y1,x2,y2,s]]}
        """

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
                v[:, -1] = (v[:, -1] - min_score) / diff

    def get_preds(self, pred_dir: str):
        events = os.listdir(pred_dir)
        boxes = dict()
        pbar = tqdm(events)

        for event in pbar:
            pbar.set_description("Reading Predictions ")
            event_dir = os.path.join(pred_dir, event)
            event_images = os.listdir(event_dir)
            current_event = dict()
            for imgtxt in event_images:
                imgname, _boxes = self.read_pred_file(os.path.join(event_dir, imgtxt))
                current_event[imgname.rstrip(".jpg")] = _boxes
            boxes[event] = current_event
        return boxes

    def read_pred_file(self, filepath):
        with open(filepath, "r") as f:
            lines = f.readlines()
            img_file = lines[0].rstrip("\n\r")
            lines = lines[2:]
        boxes = []
        for line in lines:
            line = line.rstrip("\r\n").split(" ")
            if line[0] == "":
                continue
            boxes.append(
                [
                    float(line[0]),
                    float(line[1]),
                    float(line[2]),
                    float(line[3]),
                    float(line[4]),
                ]
            )
        boxes = np.array(boxes)
        return img_file.split(os.path.sep)[-1], boxes

    def remove_cache(self):
        if os.path.exists(self.save_folder) and os.path.isdir(self.save_folder):
            shutil.rmtree(self.save_folder)
            print(f"remove '{self.save_folder}'")
        else:
            print(f"'{self.save_folder}' not exist.")
