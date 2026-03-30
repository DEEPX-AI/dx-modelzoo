import json
import tempfile
from typing import List, Tuple

import torch
from loguru import logger
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

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
        super().__init__(session, dataset, workers=12)
        self.dataset: COCOPoseDataset
        self.temp_file = None
        self.first_chunk_written = False

    def init_metrics(self) -> dict:
        """Initialize temporary file for COCO pose detections."""
        self.temp_file = tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8", suffix=".json", delete=False)
        self.temp_file.write("[\n")
        self.first_chunk_written = False
        return {"detections_count": 0}

    def extract_inputs(self, batch_data) -> torch.Tensor:
        """Extract image from batch."""
        image, origin_shape, img_id = batch_data
        return image

    def process_batch_result(
        self, batch_data: Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor], output, metrics_state: dict
    ) -> dict:
        """Process batch result and write pose detections to file."""
        image, origin_shape, img_id = batch_data
        origin_shape = [value[0] for value in origin_shape]

        boxes, scores, class_ids, keypoints = output

        if len(scores) > 0:
            if self.session.type == SessionType.dxruntime:
                image = image.permute(0, 3, 1, 2)

            scaled_boxes, scaled_keypoints = self._change_scales_to_origin(image, origin_shape, boxes, keypoints)
            scaled_boxes = convert_xyxy_2_cxcywh(scaled_boxes)
            coco_formatted_dets = self._make_coco_format_pose(scaled_boxes, scores, class_ids, scaled_keypoints, img_id)

            if coco_formatted_dets:
                json_chunk = ",\n".join([json.dumps(det) for det in coco_formatted_dets])
                if self.first_chunk_written:
                    self.temp_file.write(",\n")
                self.temp_file.write(json_chunk)
                self.first_chunk_written = True
                metrics_state["detections_count"] += len(coco_formatted_dets)

        return metrics_state

    def compute_final_metrics(self, metrics_state: dict) -> dict:
        """Compute final COCO pose metrics."""
        # Finalize JSON file
        self.temp_file.write("\n]\n")
        self.temp_file.flush()
        self.temp_file.close()

        # Run COCO evaluation
        mAP, mAP50 = self._run_coco_eval(self.temp_file.name, self.dataset.coco_annotation)

        # Clean up
        import os

        os.unlink(self.temp_file.name)

        # Calculate FPS
        avg_fps = (
            metrics_state["detections_count"] / self.total_inference_time if self.total_inference_time > 0 else 0.0
        )

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
