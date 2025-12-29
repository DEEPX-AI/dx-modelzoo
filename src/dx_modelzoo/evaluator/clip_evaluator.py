import time
from collections import deque

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from dx_modelzoo.evaluator import EvaluatorBase


class CLIPEvaluator(EvaluatorBase):
    def __init__(self, session, dataset, zero_shot_text_embedding: str) -> None:
        super().__init__(session, dataset)
        self.zero_shot_text_embedding = zero_shot_text_embedding

    def _accuracy(self, output, target, topk=(1, 5)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        pred = output.topk(max(topk), 1, True, True)[1].t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [float(correct[:k].reshape(-1).float().sum(0, keepdims=True).cpu().numpy()) for k in topk]

    def eval(self):
        loader = self.make_loader()
        total_len = len(loader)
        total_inference_time = 0.0
        recent_inference_times = deque(maxlen=50)  # total

        zeroshot_text_embedding_weight = torch.from_numpy(np.load(self.zero_shot_text_embedding))

        pbar = tqdm(loader, total=total_len)
        correct_top1 = 0
        correct_top5 = 0
        current_count = 0
        for images, labels in pbar:
            current_count += images.shape[0]
            start_time = time.time()
            image_feature = self.session.run(images)
            end_time = time.time()
            inference_time = end_time - start_time
            total_inference_time += inference_time
            recent_inference_times.append(inference_time)

            image_feature = torch.from_numpy(self.postprocessing(image_feature))
            image_feature /= image_feature.norm(dim=-1, keepdim=True)
            logits = 100.0 * image_feature @ zeroshot_text_embedding_weight

            acc1, acc5 = self._accuracy(logits, labels, topk=(1, 5))
            correct_top1 += acc1
            correct_top5 += acc5
            if len(recent_inference_times) > 0:
                current_fps = len(recent_inference_times) / sum(recent_inference_times)
            else:
                current_fps = 0.0
            pbar.desc = (
                f"ImageNet | "
                f"Top1:{correct_top1/current_count:.2f} "
                f"Top5:{correct_top5/current_count:.2f} "
                f"Current_FPS:{current_fps:.1f}"
            )
        avg_fps = total_len / total_inference_time if total_inference_time > 0 else 0
        pbar.set_postfix(
            {
                "top1": correct_top1 / total_len,
                "top5": correct_top5 / total_len,
                "inference_time": total_inference_time / total_len,
            }
        )
        print(
            f"Top1 Accuracy: {correct_top1 / total_len * 100:.2f}\n"
            f"Top5 Accuracy: {correct_top5 / total_len * 100:.2f}\n"
            f"Average FPS: {avg_fps:.2f}\n"
        )
        logger.success(
            f"@JSON <Top1 Accuracy:{correct_top1 / total_len * 100:.2f}; "
            f"Top5 Accuracy:{correct_top5 / total_len * 100:.2f}; "
            f"Average FPS:{avg_fps:.2f}>"
        )

        return {
            "performance": [correct_top1, correct_top5], 
            "fps": avg_fps,
        }