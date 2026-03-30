import math
from typing import Tuple

import numpy as np
import torch
from loguru import logger

from dx_modelzoo.dataset import DatasetBase
from dx_modelzoo.evaluator import EvaluatorBase
from dx_modelzoo.session import SessionBase


class DepthEstimationEvaluator(EvaluatorBase):
    def __init__(self, session: SessionBase, dataset: DatasetBase) -> None:
        super().__init__(session, dataset, workers=12)
        self.use_median_scaling = False

    def init_metrics(self) -> dict:
        """Initialize metrics state."""
        return {
            "rmse_sum": 0.0,
            "count": 0,
        }

    def extract_inputs(self, batch_data: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Extract images from batch data."""
        images, depth = batch_data
        return images

    def process_batch_result(
        self,
        batch_data: Tuple[torch.Tensor, torch.Tensor],
        output: np.ndarray,
        metrics_state: dict,
    ) -> dict:
        """Process batch result and update RMSE."""
        images, depth = batch_data

        output = torch.from_numpy(output)

        if self.use_median_scaling:
            gt_valid_mask = depth > 0
            scale = torch.median(depth[gt_valid_mask]) / torch.median(output[gt_valid_mask])
            output = output * scale

        valid_mask = ((depth > 0) + (output > 0)) > 0
        output_masked = output[valid_mask]
        depth_masked = depth[valid_mask]
        abs_diff = (output_masked - depth_masked).abs()

        mse = float((torch.pow(abs_diff, 2)).mean())
        rmse = math.sqrt(mse)

        metrics_state["rmse_sum"] += rmse
        metrics_state["count"] += 1

        return metrics_state

    def compute_final_metrics(self, metrics_state: dict) -> dict:
        """Compute final RMSE metric."""
        rmse_sum = metrics_state["rmse_sum"]
        count = metrics_state["count"]

        avg_rmse = rmse_sum / count if count > 0 else 0.0
        avg_fps = count / self.total_inference_time if self.total_inference_time > 0 else 0.0

        print(f"RMSE: {round(avg_rmse, 3)}")
        print(f"Average FPS: {avg_fps:.2f}")
        logger.success(f"@JSON <RMSE:{round(avg_rmse, 3)}; Average FPS:{avg_fps:.2f}>")

        return {
            "performance": [avg_rmse],
            "fps": avg_fps,
        }

    def format_progress_desc(self, metrics_state: dict, current_fps: float) -> str:
        """Format progress bar description."""
        return f"Depth | Current_FPS:{current_fps:.1f}"
