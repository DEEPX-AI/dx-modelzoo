import json
import os
import queue
import tempfile
import threading
import time
from collections import deque
from typing import List, Tuple

import torch
from loguru import logger
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from dx_modelzoo.dataset.coco_pose import COCOPoseDataset
from dx_modelzoo.enums import SessionType
from dx_modelzoo.evaluator import EvaluatorBase
from dx_modelzoo.session import SessionBase
from dx_modelzoo.utils.detection import convert_xyxy_2_cxcywh, get_pad_size, get_ratios, scale_boxes

COCO_POSE = List[int | torch.Tensor]


class COCOPoseEvaluator(EvaluatorBase):
    """COCO Pose Evaluator for obeject detection

    Args:
        session (SessionBase): runtime session.
        dataset (COCOPoseDataset): COCO pose dataset.
    """

    def __init__(self, session: SessionBase, dataset: COCOPoseDataset):
        super().__init__(session, dataset)
        self.dataset: COCOPoseDataset
        self.total_inference_time = 0.0  # Total time spent on inference
        self.recent_inference_times = deque(maxlen=50)  # total: 5000

    def eval(self) -> None:
        """evaluation OD model with COCO dataset."""
        # Use async evaluation for DX runtime
        if self.session.type == SessionType.dxruntime:
            return self._eval_async()

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
            "performance": [mAP * 100, mAP50 * 100],
            "fps": avg_fps,
        }

    def _run_one_batch(self, batch: Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]) -> str:
        image, origin_shape, img_id = batch
        origin_shape = [value[0] for value in origin_shape]

        start_time = time.time()
        model_output = self.session.run(image)  # np.ndarray
        inference_time = time.time() - start_time

        self.recent_inference_times.append(inference_time)
        self.total_inference_time += inference_time

        boxes, scores, class_ids, keypoints = self.postprocessing(model_output)

        if len(scores) == 0:
            return ""
        if self.session.type == SessionType.dxruntime:
            image = image.permute(0, 3, 1, 2)

        scaled_boxes, scaled_keypoints = self._change_scales_to_origin(image, origin_shape, boxes, keypoints)
        scaled_boxes = convert_xyxy_2_cxcywh(scaled_boxes)
        coco_formatted_dets = self._make_coco_format_pose(scaled_boxes, scores, class_ids, scaled_keypoints, img_id)

        if not coco_formatted_dets:
            return ""

        return ",\n".join([json.dumps(det) for det in coco_formatted_dets])

    def _eval_async(self) -> dict:
        """Async evaluation for DX runtime using run_async/wait pattern."""
        loader = self.make_loader()
        total_len = len(loader)

        # Shared state with locks
        lock = threading.Lock()
        file_lock = threading.Lock()
        worker_error = None

        # Queue for pending jobs: (job_id, image, origin_shape, img_id)
        pending_queue = queue.Queue(maxsize=32)
        done_event = threading.Event()

        # Wall-clock time measurement
        wall_start_time = time.time()

        pbar = tqdm(total=total_len, desc="COCO")

        with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8", suffix=".json", delete=False) as temp_f:
            temp_file_path = temp_f.name
            temp_f.write("[\n")
            first_chunk_written = [False]  # Use list for mutability in closure

            def result_worker():
                """Worker thread that collects results and writes to JSON."""
                nonlocal worker_error

                while True:
                    try:
                        item = pending_queue.get(timeout=0.1)
                    except queue.Empty:
                        if done_event.is_set():
                            break
                        continue

                    if item is None:  # Sentinel
                        break

                    job_id, image, origin_shape, img_id = item

                    try:
                        # Wait for NPU result
                        model_output = self.session.wait(job_id)

                        # Postprocessing
                        boxes, scores, class_ids, keypoints = self.postprocessing(model_output)

                        json_chunk = ""
                        if len(scores) > 0:
                            # For dxruntime, image is already NHWC, need to permute for scale calculation
                            image_for_scale = image.permute(0, 3, 1, 2)

                            scaled_boxes, scaled_keypoints = self._change_scales_to_origin(
                                image_for_scale, origin_shape, boxes, keypoints
                            )
                            scaled_boxes = convert_xyxy_2_cxcywh(scaled_boxes)
                            coco_formatted_dets = self._make_coco_format_pose(
                                scaled_boxes, scores, class_ids, scaled_keypoints, img_id
                            )

                            if coco_formatted_dets:
                                json_chunk = ",\n".join([json.dumps(det) for det in coco_formatted_dets])

                        # Write to file (thread-safe)
                        if json_chunk:
                            with file_lock:
                                if first_chunk_written[0]:
                                    temp_f.write(",\n")
                                temp_f.write(json_chunk)
                                first_chunk_written[0] = True
                                temp_f.flush()

                        with lock:
                            # Wall-clock based FPS
                            elapsed = time.time() - wall_start_time
                            processed_count = pbar.n + 1
                            current_fps = processed_count / elapsed if elapsed > 0 else 0.0

                            pbar.set_description(f"COCO | Current_FPS:{current_fps:.1f}")
                            pbar.update(1)

                    except Exception as e:
                        with lock:
                            worker_error = e
                        break

            # Start worker thread
            worker_thread = threading.Thread(target=result_worker, daemon=True)
            worker_thread.start()

            try:
                # Main thread: submit jobs
                for batch in loader:
                    with lock:
                        if worker_error is not None:
                            raise worker_error

                    image, origin_shape, img_id = batch
                    origin_shape_processed = [value[0] for value in origin_shape]

                    job_id = self.session.run_async(image)

                    pending_queue.put((job_id, image, origin_shape_processed, img_id))

                # Signal completion and wait for worker
                done_event.set()
                pending_queue.put(None)  # Sentinel
                worker_thread.join()

                if worker_error is not None:
                    raise worker_error

            finally:
                pbar.close()

            temp_f.write("\n]\n")
            temp_f.flush()

            mAP, mAP50 = self._run_coco_eval(temp_file_path, self.dataset.coco_annotation)

        # Clean up temp file
        os.unlink(temp_file_path)

        total_wall_time = time.time() - wall_start_time
        avg_fps = total_len / total_wall_time if total_wall_time > 0 else 0

        print(f"mAP: {round(mAP*100, 3)} mAP50: {round(mAP50*100, 3)}")
        print(f"Average Inference FPS: {avg_fps:.2f}")
        logger.success(f"@JSON <mAP:{round(mAP*100, 3)}; mAP50:{round(mAP50*100, 3)}; Average FPS:{avg_fps:.2f}>")

        return {
            "performance": [mAP * 100, mAP50 * 100],
            "fps": avg_fps,
        }

    def _change_scales_to_origin(
        self, image: torch.Tensor, origin_shape: List[torch.Tensor], boxes: torch.Tensor, keypoints: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cloned_boxes = boxes.clone()
        ratios = get_ratios(image, origin_shape)
        pads = get_pad_size(image, origin_shape, ratios)

        # Box scaling using scale_boxes (same as object detection)
        cloned_boxes = scale_boxes(cloned_boxes, origin_shape, ratios, pads)

        # Keypoint scaling (manual process)
        cloned_keypoints = keypoints.clone()

        # Remove padding first
        x_pad = pads[1]
        y_pad = pads[0]

        cloned_keypoints[..., 0] -= x_pad
        cloned_keypoints[..., 1] -= y_pad

        cloned_keypoints[..., 0] /= ratios[1]
        cloned_keypoints[..., 1] /= ratios[0]

        cloned_keypoints[..., 0] = cloned_keypoints[..., 0].clamp(0, origin_shape[1])
        cloned_keypoints[..., 1] = cloned_keypoints[..., 1].clamp(0, origin_shape[0])

        return cloned_boxes, cloned_keypoints

    def _make_coco_format_pose(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        class_ids: torch.Tensor,
        keypoints: torch.Tensor,
        img_id: torch.Tensor,
    ) -> str:
        detections = []
        boxes_list = boxes.tolist()
        scores_list = scores.tolist()
        image_id = int(img_id.item())

        detections = []
        for i in range(len(scores)):
            detection = {
                "image_id": image_id,
                "category_id": 1,  # COCO person category_id is 1
                "bbox": [round(coord, 3) for coord in boxes_list[i]],
                "score": round(scores_list[i], 5),
                "keypoints": [round(coord, 3) for coord in keypoints[i].flatten().tolist()],
            }
            detections.append(detection)

        return detections

    def _run_coco_eval(self, result_file_path: str, coco_annotation: COCO) -> None:
        """run coco evaluation.

        Args:
            coco_det_list (List[List]): coco det list.
            coco_annotation (COCO): coco annoation.
        """
        # Guard: empty results file → return zeros
        with open(result_file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception:
                data = []
        if not data:
            return 0.0, 0.0

        predicted_det = coco_annotation.loadRes(result_file_path)

        coco_eval = COCOeval(coco_annotation, predicted_det, "keypoints")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        mAP, mAP50 = coco_eval.stats[:2]
        return mAP, mAP50
