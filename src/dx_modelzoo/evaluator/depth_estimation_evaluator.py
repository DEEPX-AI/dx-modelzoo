import math
import time
from collections import deque

import torch
from loguru import logger
from tqdm import tqdm

from dx_modelzoo.evaluator import EvaluatorBase


class DepthEstimationEvaluator(EvaluatorBase):
    def __init__(self, session, dataset) -> None:
        super().__init__(session, dataset)

    def eval(self):
        loader = self.make_loader()
        total_len = len(loader)
        total_inference_time = 0.0
        recent_inference_times = deque(maxlen=50)  # total

        rmse_sum = 0
        pbar = tqdm(loader, total=total_len)
        for images, depth in pbar:
            start_time = time.time()
            output = self.session.run(images)
            inference_time = time.time() - start_time

            recent_inference_times.append(inference_time)
            total_inference_time += inference_time

            output = self.postprocessing(output)

            if len(recent_inference_times) > 0:
                current_fps = len(recent_inference_times) / sum(recent_inference_times)
            else:
                current_fps = 0.0

            pbar.desc = f"Cuurent_FPS:{current_fps:.1f}"
            output = torch.from_numpy(output)
            valid_mask = ((depth > 0) + (output > 0)) > 0
            output = output[valid_mask]
            depth = depth[valid_mask]
            abs_diff = (output - depth).abs()

            mse = float((torch.pow(abs_diff, 2)).mean())
            rmse_sum += math.sqrt(mse)
        avg_fps = total_len / total_inference_time if total_inference_time > 0 else 0.0
        print(f"RMSE: {round(rmse_sum / total_len, 3)}")
        print(f"Average FPS: {avg_fps:.2f}")
        logger.success(f"@JSON <RMSE:{round(rmse_sum / total_len, 3)}; Average FPS:{avg_fps:.2f}>")

        return {
            "performance": [rmse_sum / total_len], 
            "fps": avg_fps,
        }
