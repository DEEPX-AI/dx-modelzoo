from typing import List
from collections import deque

import numpy as np
import torch
from tqdm import tqdm

from dx_modelzoo.dataset import DatasetBase
from dx_modelzoo.evaluator import EvaluatorBase
from dx_modelzoo.session import SessionBase
from dx_modelzoo.utils import torch_to_numpy

from loguru import logger
import time


class ICEvaluator(EvaluatorBase):
    """Image Classification Evaluator."""

    def __init__(self, session: SessionBase, dataset: DatasetBase) -> None:
        super().__init__(session, dataset)
        self.total_inference_time = 0.0  # Total time spent on inference
        self.recent_inference_times = deque(maxlen=200)  # total:50000
        
    def eval(self) -> None:
        """evaluation IC Model."""
        loader = self.make_loader()
        total_len = len(loader)
        
        topk_correct_count = [0, 0]
        current_count = 0

        pbar = tqdm(enumerate(loader), total=total_len)
        for batch, (image, label) in pbar:
            batch_size = image.size()[0]
            current_count += batch_size

            correct = self._run_one_batch(image, label)
            topk_correct_count = self._topk_eval(topk_correct_count, correct)

            if len(self.recent_inference_times) > 0:
                current_fps = 200 / sum(self.recent_inference_times)
            else:
                current_fps = 0.0
                
            pbar.desc = (
                f"ImageNet | "
                f"Top1:{topk_correct_count[0]/current_count:.2f} "
                f"Top5:{topk_correct_count[1]/current_count:.2f} "
                f"Current_FPS:{current_fps:.1f}"
            )
        
        # Calculate final metrics
        top1_acc = (topk_correct_count[0]/current_count)*100
        top5_acc = (topk_correct_count[1]/current_count)*100
        avg_fps = current_count / self.total_inference_time if self.total_inference_time > 0 else 0

         # Print and log results
        print(
            f"Top1 Accuracy: {top1_acc:.2f}\n"
            f"Top5 Accuracy:  {top5_acc:.2f}\n"
            f"Average FPS: {avg_fps:.2f}") 
        logger.success(f"@JSON <Top1 Accuracy:{top1_acc:.2f}; Top5 Accuracy:{top5_acc:.2f}; Average FPS:{avg_fps:.2f}>")

    def _run_one_batch(self, image: torch.Tensor, label: torch.Tensor) -> np.ndarray:
        """run one batch.

        Args:
            image (torch.Tensor): batch image.
            label (torch.Tensor): batch label.

        Returns:
            np.ndarray: run output.
        """
        start_time = time.time()
        output = self.session.run(image)
        inference_time = time.time() - start_time
            
        self.recent_inference_times.append(inference_time)     
        self.total_inference_time += inference_time

        output = self.postprocessing(output)
        label = torch_to_numpy(label)
        label = np.reshape(label, [-1, 1])
        return np.equal(output, label)

    def _topk_eval(self, topk_correct_count: List[int], correct: np.ndarray, topk=[1, 5]) -> List[int]:
        """topk evaluation.

        Args:
            topk_correct_count (List[int]): topk correct count list.
            correct (np.ndarray): correct.
            topk (list, optional): topk value.. Defaults to [1, 5].

        Returns:
            List[int]: updated topk correct count list.
        """
        for idx_k, k in enumerate(topk):
            topk_correct_count[idx_k] += np.sum(correct[..., :k])
        return topk_correct_count
