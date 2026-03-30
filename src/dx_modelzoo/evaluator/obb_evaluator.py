from typing import Any, List, Tuple

import cv2
import numpy as np
import torch
from loguru import logger

from dx_modelzoo.dataset.dotav1 import DOTAV1Dataset
from dx_modelzoo.evaluator import EvaluatorBase
from dx_modelzoo.models.object_detection.nms import batch_probiou
from dx_modelzoo.session import SessionBase
from dx_modelzoo.utils.detection import get_pad_size, get_ratios


class OBBEvaluator(EvaluatorBase):
    """OBB (Oriented Bounding Box) Evaluator for DOTA dataset.

    Uses Ultralytics-style metrics calculation with batch_probiou for IoU computation.
    Based on ultralytics.models.yolo.obb.val.OBBValidator.
    """

    def __init__(
        self,
        session: SessionBase,
        dataset: DOTAV1Dataset,
        save_visualizations: bool = False,
        vis_output_dir: str = None,
        max_vis_images: int = 10,
    ):
        super().__init__(session, dataset, workers=12)
        self.dataset: DOTAV1Dataset

        # IoU thresholds for mAP calculation (Ultralytics style)
        self.iouv = torch.linspace(0.5, 0.95, 10)  # mAP@0.5:0.95
        self.niou = self.iouv.numel()

        # Stats storage (Ultralytics style)
        self.stats = []

        # Visualization settings
        self.save_visualizations = save_visualizations
        self.vis_output_dir = vis_output_dir or "./obb_visualizations"
        self.max_vis_images = max_vis_images
        self.vis_count = 0

        if self.save_visualizations:
            import os

            os.makedirs(self.vis_output_dir, exist_ok=True)
            logger.info(f"Saving OBB visualizations to: {self.vis_output_dir}")

    def init_metrics(self) -> dict:
        """Initialize metrics storage."""
        self.stats = []
        return {"processed_images": 0}

    def extract_inputs(self, batch_data: Tuple[torch.Tensor, List[torch.Tensor], str]) -> torch.Tensor:
        """Extract image from batch data."""
        image, origin_shape, img_id = batch_data
        return image

    def process_batch_result(
        self,
        batch_data: Tuple[torch.Tensor, List[torch.Tensor], str],
        output: Any,
        metrics_state: dict,
    ) -> dict:
        """Process batch result using Ultralytics approach.

        Follows ultralytics.models.yolo.detect.val.DetectionValidator.update_metrics
        """
        image, origin_shape, img_id = batch_data
        origin_shape = [value[0] for value in origin_shape]

        # For dxruntime format
        # if hasattr(image, "permute"):
        #     image = image.permute(0, 3, 1, 2)
        if image.shape[-1] in [1, 3]:
            image = image.permute(0, 3, 1, 2)

        # Prepare predictions (scale to original image)
        pred = self._prepare_pred(output, image, origin_shape)

        # Prepare batch ground truth (using origin_shape)
        gt_batch = self._prepare_batch(img_id, origin_shape)

        # Compute statistics using Ultralytics method
        stat = self._process_batch(pred, gt_batch)
        self.stats.append(stat)

        # Save visualization if enabled
        if self.save_visualizations and self.vis_count < self.max_vis_images:
            self._save_visualization(image, pred, gt_batch, img_id, origin_shape)
            self.vis_count += 1

        metrics_state["processed_images"] += 1
        return metrics_state

    def compute_final_metrics(self, metrics_state: dict) -> dict:
        """Compute final metrics using Ultralytics ap_per_class function."""
        from dx_modelzoo.dataset.dotav1 import DOTA_CLASSES

        # Gather stats (Ultralytics style)
        if not self.stats:
            logger.warning("No predictions to evaluate")
            return {"performance": [0.0, 0.0], "fps": 0.0}

        # Concatenate all stats
        stats_dict = {}
        for k in self.stats[0].keys():
            stats_dict[k] = np.concatenate([s[k] for s in self.stats], 0)

        # Debug: Print stats summary
        print(f"Total predictions: {len(stats_dict['pred_cls'])}")
        print(f"Total targets: {len(stats_dict['target_cls'])}")
        print(f"True positives (any IoU): {stats_dict['tp'].any(axis=1).sum()}")
        print(f"True positives (IoU@0.5): {stats_dict['tp'][:, 0].sum()}")

        # Calculate metrics using Ultralytics ap_per_class
        if len(stats_dict["tp"]):
            results = self._ap_per_class(
                stats_dict["tp"],
                stats_dict["conf"],
                stats_dict["pred_cls"],
                stats_dict["target_cls"],
                names={i: name for i, name in enumerate(DOTA_CLASSES)},
            )
            # results: (nt, fp, p, r, f1, ap, unique_classes)
            nt, fp, p, r, f1, ap, unique_classes = results

            # ap shape: (nc, niou) where nc = num classes, niou = 10
            # ap[:, 0] = AP@IoU=0.5
            # ap.mean(0) = mAP per IoU threshold across all classes
            # ap.mean(1) = mAP per class across all IoU thresholds

            if ap.size > 0:
                map50 = ap[:, 0].mean()  # mAP@0.5
                map75 = ap[:, 5].mean() if ap.shape[1] > 5 else 0.0  # mAP@0.75
                map_avg = ap.mean()  # mAP@0.5:0.95
                # mp = p.mean()  # mean precision
                # mr = r.mean()  # mean recall

                # Print per-class AP for debugging
                print("Per-class AP@0.5:")
                for i, cls_idx in enumerate(unique_classes):
                    cls_name = DOTA_CLASSES[cls_idx] if cls_idx < len(DOTA_CLASSES) else f"class{cls_idx}"
                    print(f"  {cls_name:20s}: AP50={ap[i, 0]:.3f}, AP={ap[i, :].mean():.3f}, n={nt[i]:5d}")
            else:
                map50 = map75 = map_avg = 0.0
        else:
            map50 = map75 = map_avg = 0.0

        # Calculate FPS
        avg_fps = (
            metrics_state["processed_images"] / self.total_inference_time if self.total_inference_time > 0 else 0.0
        )

        # Print results (COCO style)
        print("")
        print(f" Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {map_avg:.3f}")
        print(f" Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {map50:.3f}")
        print(f" Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {map75:.3f}")
        print(f"mAP: {round(map_avg*100, 3)} mAP50: {round(map50*100, 3)}")
        print(f"Average Inference FPS: {avg_fps:.2f}")
        logger.success(f"@JSON <mAP:{round(map_avg*100, 3)}; mAP50:{round(map50*100, 3)}; Average FPS:{avg_fps:.2f}>")

        return {
            "performance": [map_avg * 100, map50 * 100],
            "fps": avg_fps,
        }

    def format_progress_desc(self, metrics_state: dict, current_fps: float) -> str:
        """Format progress bar description."""
        img_count = metrics_state.get("processed_images", 0)
        return f"OBB | Images:{img_count} Current_FPS:{current_fps:.1f}"

    def _prepare_pred(self, output: torch.Tensor, image: torch.Tensor, origin_shape: List[int]) -> dict:
        """Prepare predictions in Ultralytics format.

        Returns:
            dict with keys: 'bboxes' (N, 5), 'conf' (N,), 'cls' (N,)
        """
        if output is None or len(output) == 0:
            return {"bboxes": torch.zeros((0, 5)), "conf": torch.zeros(0), "cls": torch.zeros(0)}

        # Scale boxes to original image size
        scaled_output = self._change_box_scales_to_origin(image, origin_shape, output)

        # Convert to Ultralytics format
        # Model output: [cx, cy, w, h, conf, class, angle]
        bboxes = torch.cat([scaled_output[:, :4], scaled_output[:, 6:7]], dim=1)  # [cx, cy, w, h, angle]
        conf = scaled_output[:, 4]
        cls = scaled_output[:, 5]

        return {"bboxes": bboxes, "conf": conf, "cls": cls}

    def _prepare_batch(self, img_id: str, origin_shape: List[int]) -> dict:
        """Prepare ground truth batch in Ultralytics format.

        Args:
            img_id: Image identifier
            origin_shape: Original image shape [H, W]

        Returns:
            dict with keys: 'bboxes' (M, 5), 'cls' (M,)
        """
        # Get ground truth for this image
        img_id = str(img_id[0] if isinstance(img_id, (list, tuple)) else img_id)

        label_path = f"{self.dataset.label_dir}/{img_id}.txt"

        bboxes_list = []
        cls_list = []

        # Extract origin_shape dimensions
        # origin_shape can be [H, W] or [[H], [W]] or [Tensor(H), Tensor(W)]
        if isinstance(origin_shape[0], (list, tuple)):
            img_h = origin_shape[0][0] if isinstance(origin_shape[0][0], (int, float)) else origin_shape[0][0].item()
            img_w = origin_shape[1][0] if isinstance(origin_shape[1][0], (int, float)) else origin_shape[1][0].item()
        else:
            # Handle Tensor or scalar values
            img_h = origin_shape[0].item() if hasattr(origin_shape[0], "item") else origin_shape[0]
            img_w = origin_shape[1].item() if hasattr(origin_shape[1], "item") else origin_shape[1]

        try:
            with open(label_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 9:
                        try:
                            class_idx = int(parts[0])
                            # Normalized coordinates [x1, y1, x2, y2, x3, y3, x4, y4]
                            coords = [float(p) for p in parts[1:9]]

                            # Convert to absolute coordinates using ORIGIN image size
                            # DOTA labels are normalized to [0, 1] based on original image size
                            x_coords = np.array(coords[0::2]) * img_w
                            y_coords = np.array(coords[1::2]) * img_h

                            # Convert to xywhr using cv2.minAreaRect
                            points = np.array([[x_coords[i], y_coords[i]] for i in range(4)], dtype=np.float32)
                            (cx, cy), (w, h), angle_deg = cv2.minAreaRect(points)

                            # Convert to Ultralytics format: angle in [-π/4, 3π/4), w >= h
                            angle = np.deg2rad(angle_deg)
                            if w < h:
                                w, h = h, w
                                angle += np.pi / 2
                            while angle >= 3 * np.pi / 4:
                                angle -= np.pi
                            while angle < -np.pi / 4:
                                angle += np.pi

                            bboxes_list.append([cx, cy, w, h, angle])
                            cls_list.append(class_idx)
                        except (ValueError, IndexError) as e:
                            if not hasattr(self, "_logged_parse_error"):
                                self._logged_parse_error = True
                                logger.warning(f"Parse error: {e}, line: {line.strip()}")
                            continue
        except FileNotFoundError:
            if not hasattr(self, "_logged_file_not_found"):
                self._logged_file_not_found = True
                logger.warning(f"Label file not found: {label_path}")
            pass

        if len(bboxes_list) == 0:
            return {"bboxes": torch.zeros((0, 5)), "cls": torch.zeros(0)}

        return {
            "bboxes": torch.tensor(bboxes_list, dtype=torch.float32),
            "cls": torch.tensor(cls_list, dtype=torch.float32),
        }

    def _process_batch(self, preds: dict, batch: dict) -> dict:
        """Process batch using Ultralytics method with batch_probiou.

        Based on ultralytics.models.yolo.obb.val.OBBValidator._process_batch
        """
        if batch["cls"].shape[0] == 0 or preds["cls"].shape[0] == 0:
            return {
                "tp": np.zeros((preds["cls"].shape[0], self.niou), dtype=bool),
                "conf": preds["conf"].cpu().numpy(),
                "pred_cls": preds["cls"].cpu().numpy(),
                "target_cls": batch["cls"].cpu().numpy(),
            }

        # Compute IoU using batch_probiou (Ultralytics method)
        iou = batch_probiou(batch["bboxes"], preds["bboxes"])

        # Match predictions using Ultralytics method
        correct = self._match_predictions(preds["cls"], batch["cls"], iou)

        return {
            "tp": correct.cpu().numpy(),
            "conf": preds["conf"].cpu().numpy(),
            "pred_cls": preds["cls"].cpu().numpy(),
            "target_cls": batch["cls"].cpu().numpy(),
        }

    def _match_predictions(
        self, pred_classes: torch.Tensor, true_classes: torch.Tensor, iou: torch.Tensor
    ) -> torch.Tensor:
        """Match predictions to ground truth using IoU (Ultralytics method).

        Based on ultralytics.engine.validator.BaseValidator.match_predictions

        Returns:
            correct (torch.Tensor): Shape (N, 10) for 10 IoU thresholds
        """
        # Dx10 matrix, where D - detections, 10 - IoU thresholds
        correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)
        # LxD matrix where L - labels (rows), D - detections (columns)
        correct_class = true_classes[:, None] == pred_classes
        iou = iou * correct_class  # zero out the wrong classes
        iou = iou.cpu().numpy()

        for i, threshold in enumerate(self.iouv.cpu().tolist()):
            matches = np.nonzero(iou >= threshold)  # IoU > threshold and classes match
            matches = np.array(matches).T
            if matches.shape[0]:
                if matches.shape[0] > 1:
                    matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True

        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)

    def _change_box_scales_to_origin(
        self, image: torch.Tensor, origin_shape: List[int], outputs: torch.Tensor
    ) -> torch.Tensor:
        """Scale OBB boxes to original image size.

        Args:
            image: Preprocessed image
            origin_shape: Original image shape [H, W]
            outputs: Model outputs [cx, cy, w, h, conf, class, angle]

        Returns:
            Scaled outputs with normalized angle
        """
        if len(outputs) == 0:
            return outputs

        use_both_ratios = getattr(self, "use_both_ratios", False)
        use_padding = getattr(self, "use_padding", True)

        cloned = outputs.clone()
        ratios = get_ratios(image, origin_shape, use_both_ratios)
        pads = get_pad_size(image, origin_shape, ratios)

        # Scale cx, cy, w, h (indices 0-3)
        if use_padding:
            cloned[:, 0] -= pads[1]  # cx
            cloned[:, 1] -= pads[0]  # cy

        cloned[:, 0] /= ratios[1]  # cx
        cloned[:, 1] /= ratios[0]  # cy
        cloned[:, 2] /= ratios[1]  # w
        cloned[:, 3] /= ratios[0]  # h

        # Clip to image bounds
        cloned[:, 0].clamp_(0, origin_shape[1])
        cloned[:, 1].clamp_(0, origin_shape[0])

        # Normalize angle to Ultralytics format (index 6)
        if cloned.shape[1] > 6:
            angle = cloned[:, 6]
            w = cloned[:, 2]
            h = cloned[:, 3]

            # Ensure w >= h
            mask = w < h
            if mask.any():
                cloned[mask, 2], cloned[mask, 3] = cloned[mask, 3].clone(), cloned[mask, 2].clone()
                angle[mask] += np.pi / 2

            # Normalize to [-π/4, 3π/4)
            while (angle >= 3 * np.pi / 4).any():
                angle[angle >= 3 * np.pi / 4] -= np.pi
            while (angle < -np.pi / 4).any():
                angle[angle < -np.pi / 4] += np.pi

            cloned[:, 6] = angle

        return cloned

    def _ap_per_class(
        self, tp: np.ndarray, conf: np.ndarray, pred_cls: np.ndarray, target_cls: np.ndarray, names: dict = {}
    ) -> tuple:
        """Compute AP per class (simplified Ultralytics version).

        Based on ultralytics.utils.metrics.ap_per_class
        """
        # Sort by confidence
        i = np.argsort(-conf)
        tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

        # Find unique classes
        unique_classes, nt = np.unique(target_cls, return_counts=True)
        nc = unique_classes.shape[0]

        # Create AP array
        ap = np.zeros((nc, tp.shape[1]))
        p = np.zeros(nc)
        r = np.zeros(nc)

        for ci, c in enumerate(unique_classes):
            i = pred_cls == c
            n_l = nt[ci]  # number of labels
            n_p = i.sum()  # number of predictions

            if n_p == 0 or n_l == 0:
                continue

            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + 1e-16)
            r[ci] = recall[:, 0].max() if recall.shape[0] else 0.0

            # Precision
            precision = tpc / (tpc + fpc)
            p[ci] = precision[:, 0].max() if precision.shape[0] else 0.0

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j] = self._compute_ap(recall[:, j], precision[:, j])

        # Compute F1 score
        f1 = 2 * p * r / (p + r + 1e-16)

        return (nt.astype(int), np.zeros(nc), p, r, f1, ap, unique_classes.astype(int))

    def _compute_ap(self, recall: np.ndarray, precision: np.ndarray) -> float:
        """Compute AP using 101-point interpolation (Ultralytics method)."""
        # Append sentinel values
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))

        # Compute precision envelope
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

        # Integrate with 101-point interpolation
        x = np.linspace(0, 1, 101)
        ap = np.trapz(np.interp(x, mrec, mpre), x)

        return ap

    def _save_visualization(
        self, image: torch.Tensor, pred: dict, gt_batch: dict, img_id: str, origin_shape: List[int]
    ):
        """Save OBB visualization with predictions and ground truth.

        Args:
            image: Preprocessed image tensor (C, H, W)
            pred: Prediction dict with 'bboxes', 'conf', 'cls'
            gt_batch: GT dict with 'bboxes', 'cls'
            img_id: Image ID
            origin_shape: Original image shape [H, W, C]
        """
        import os

        from dx_modelzoo.dataset.dotav1 import DOTA_CLASSES

        # Get image ID
        img_id_str = str(img_id[0] if isinstance(img_id, (list, tuple)) else img_id)

        # Load original image
        img_path = f"{self.dataset.image_dir}/{img_id_str}.png"
        if not os.path.exists(img_path):
            return

        img = cv2.imread(img_path)
        if img is None:
            return

        # Draw GT boxes in green
        for i in range(len(gt_batch["bboxes"])):
            bbox = gt_batch["bboxes"][i].cpu().numpy()
            cls_idx = int(gt_batch["cls"][i].item())
            cls_name = DOTA_CLASSES[cls_idx] if cls_idx < len(DOTA_CLASSES) else f"cls{cls_idx}"

            # Convert xywhr to 4 corner points
            cx, cy, w, h, angle = bbox
            box_pts = self._xywhr_to_corners(cx, cy, w, h, angle)
            box_pts = box_pts.astype(np.int32)

            # Draw rotated box
            cv2.polylines(img, [box_pts], True, (0, 255, 0), 2)  # Green for GT
            cv2.putText(img, f"GT:{cls_name}", (int(cx), int(cy - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw prediction boxes in red
        for i in range(len(pred["bboxes"])):
            bbox = pred["bboxes"][i].cpu().numpy()
            conf = pred["conf"][i].item()
            cls_idx = int(pred["cls"][i].item())
            cls_name = DOTA_CLASSES[cls_idx] if cls_idx < len(DOTA_CLASSES) else f"cls{cls_idx}"

            # Convert xywhr to 4 corner points
            cx, cy, w, h, angle = bbox
            box_pts = self._xywhr_to_corners(cx, cy, w, h, angle)
            box_pts = box_pts.astype(np.int32)

            # Draw rotated box
            cv2.polylines(img, [box_pts], True, (0, 0, 255), 2)  # Red for predictions
            cv2.putText(
                img, f"{cls_name}:{conf:.2f}", (int(cx), int(cy + 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
            )

        # Save image
        output_path = os.path.join(self.vis_output_dir, f"{img_id_str}_obb.jpg")
        cv2.imwrite(output_path, img)

    def _xywhr_to_corners(self, cx: float, cy: float, w: float, h: float, angle: float) -> np.ndarray:
        """Convert xywhr format to 4 corner points.

        Args:
            cx, cy: Center coordinates
            w, h: Width and height
            angle: Rotation angle in radians

        Returns:
            corners: (4, 2) array of corner points
        """
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        # Half dimensions
        w_half = w / 2
        h_half = h / 2

        # Corners in local coordinates (before rotation)
        corners_local = np.array([[-w_half, -h_half], [w_half, -h_half], [w_half, h_half], [-w_half, h_half]])

        # Rotation matrix
        rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

        # Rotate and translate
        corners = corners_local @ rot_matrix.T + np.array([cx, cy])

        return corners

        # Rotation matrix
        rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

        # Rotate and translate
        corners = corners_local @ rot_matrix.T + np.array([cx, cy])

        return corners
