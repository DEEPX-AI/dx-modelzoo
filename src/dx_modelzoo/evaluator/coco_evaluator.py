import json
import tempfile
from typing import Any, List, Tuple

import torch
from loguru import logger
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from dx_modelzoo.dataset.coco import COCODataset
from dx_modelzoo.enums import SessionType
from dx_modelzoo.evaluator import EvaluatorBase
from dx_modelzoo.evaluator.constant import COCO80TO91MAPPER
from dx_modelzoo.models.object_detection.ppu import yolo_ppu_postprocessing
from dx_modelzoo.session import SessionBase
from dx_modelzoo.utils.detection import convert_xyxy_2_cxcywh, get_pad_size, get_ratios, scale_boxes

COCO_DET = List[int | torch.Tensor]


class COCOEvaluator(EvaluatorBase):
    """COCO Evaluator for object detection.

    Note: COCO evaluation requires writing detections to a JSON file,
    so this evaluator overrides the base eval() method instead of using
    the standard abstract method pattern.
    """

    def __init__(
        self,
        session: SessionBase,
        dataset: COCODataset,
    ):
        super().__init__(session, dataset, workers=12)
        self.dataset: COCODataset
        self.temp_file = None
        self.first_chunk_written = False

    def init_metrics(self) -> dict:
        """Initialize temporary file for COCO detections."""
        self.temp_file = tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8", suffix=".json", delete=False)
        self.temp_file.write("[\n")
        self.first_chunk_written = False
        return {"detections_count": 0}

    def extract_inputs(self, batch_data: Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """Extract image from batch data."""
        image, origin_shape, img_id = batch_data
        return image

    def process_batch_result(
        self,
        batch_data: Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor],
        output: Any,
        metrics_state: dict,
    ) -> dict:
        """Process batch result and write to temporary JSON file."""
        image, origin_shape, img_id = batch_data
        origin_shape = [value[0] for value in origin_shape]

        if getattr(self, "use_ppu", False):
            if self.session.type == SessionType.dxruntime:
                output = yolo_ppu_postprocessing(
                    output,
                    box_format=getattr(self, "box_format", "xyxy"),
                    anchors=getattr(self, "anchors", None),
                    strides=getattr(self, "strides", None),
                    yolo_version=getattr(self, "yolo_version", None),
                )
            else:
                output = self.postprocessing(output)

        # For dxruntime format
        if image.shape[-1] in [1, 3]:
            image = image.permute(0, 3, 1, 2)

        scaled_boxes = self._change_box_scales_to_origin(image, origin_shape, output)
        scaled_boxes = convert_xyxy_2_cxcywh(scaled_boxes)
        coco_formatted_dets = self._make_coco_format_det(scaled_boxes, output, img_id)

        # Write to file
        if coco_formatted_dets:
            json_chunk = ",\n".join([json.dumps(det) for det in coco_formatted_dets])
            if self.first_chunk_written:
                self.temp_file.write(",\n")
            self.temp_file.write(json_chunk)
            self.first_chunk_written = True
            metrics_state["detections_count"] += len(coco_formatted_dets)

        return metrics_state

    def compute_final_metrics(self, metrics_state: dict) -> dict:
        """Compute final COCO metrics from temporary file."""
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
        """Format progress bar description."""
        det_count = metrics_state.get("detections_count", 0)
        return f"COCO | Dets:{det_count} Current_FPS:{current_fps:.1f}"

    def _change_box_scales_to_origin(
        self, image: torch.Tensor, origin_shape: List[torch.Tensor], outputs: torch.Tensor
    ) -> torch.Tensor:
        """Change output bounding boxes scales to origin image.

        Args:
            image: Preprocessed image tensor
            origin_shape: Original image shape
            outputs: Model outputs (boxes in xyxy format)

        Returns:
            Scaled boxes in xyxy format
        """
        use_both_ratios = getattr(self, "use_both_ratios", False)
        use_padding = getattr(self, "use_padding", True)

        cloned = outputs.clone()
        ratios = get_ratios(image, origin_shape, use_both_ratios)
        pads = get_pad_size(image, origin_shape, ratios)

        return scale_boxes(cloned[..., :4], origin_shape, ratios, pads, use_padding)

    def _make_coco_format_det(self, boxes: torch.Tensor, outputs: torch.Tensor, img_id: torch.Tensor) -> List[COCO_DET]:
        """Make COCO format detections.

        Args:
            boxes: Scaled boxes in cxcywh format
            outputs: Model outputs
            img_id: COCO image ID

        Returns:
            List of COCO format detections
        """
        detections = []
        boxes_list = boxes.cpu().numpy().tolist()
        outputs_list = outputs.cpu().numpy().tolist()
        image_id = int(img_id.item())

        for box, output in zip(boxes_list, outputs_list):
            detection = {
                "image_id": image_id,
                "category_id": int(output[5])
                if getattr(self, "use_class_90", False)
                else COCO80TO91MAPPER[int(output[5])],
                "bbox": [round(coord, 3) for coord in box],
                "score": round(output[4], 5),
            }
            detections.append(detection)
        return detections

    def _run_coco_eval(self, result_file_path: str, coco_annotation: COCO) -> Tuple[float, float]:
        """Run COCO evaluation.

        Args:
            result_file_path: Path to detection results JSON file
            coco_annotation: COCO ground truth annotations

        Returns:
            Tuple of (mAP, mAP50)
        """
        predicted_det = coco_annotation.loadRes(result_file_path)

        coco_eval = COCOeval(coco_annotation, predicted_det, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        mAP, mAP50 = coco_eval.stats[:2]
        return mAP, mAP50
