import queue
import threading
import time
from collections import deque

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from dx_modelzoo.enums import SessionType
from dx_modelzoo.evaluator import EvaluatorBase


class CLIPEvaluator(EvaluatorBase):
    def __init__(
        self,
        session,
        dataset,
        zero_shot_text_embedding: str,
        async_queue_size: int = 32,
    ) -> None:
        super().__init__(session, dataset)
        self.zero_shot_text_embedding = zero_shot_text_embedding
        self.async_queue_size = async_queue_size
        self._use_async = session.type == SessionType.dxruntime

    def _accuracy(self, output, target, topk=(1, 5)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        pred = output.topk(max(topk), 1, True, True)[1].t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [float(correct[:k].reshape(-1).float().sum(0, keepdims=True).cpu().numpy()) for k in topk]

    def eval(self):
        if self._use_async:
            logger.info(f"Using async evaluation with queue_size={self.async_queue_size}")
            return self._eval_async()
        return self._eval_sync()

    def _eval_sync(self):
        """Synchronous evaluation (original behavior)."""
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

        return self._finalize_results(correct_top1, correct_top5, current_count, total_inference_time)

    def _eval_async(self):
        """Asynchronous evaluation for NPU session using native async API."""
        loader = self.make_loader()
        total_len = len(loader)

        # Load text embedding weight (shared across threads)
        zeroshot_text_embedding_weight = torch.from_numpy(np.load(self.zero_shot_text_embedding))

        # Thread-safe state
        job_queue: queue.Queue = queue.Queue(maxsize=self.async_queue_size)
        result_lock = threading.Lock()

        correct_top1 = 0
        correct_top5 = 0
        current_count = 0
        worker_done = threading.Event()

        wall_start_time = time.time()
        pbar = tqdm(total=total_len)

        def worker_thread():
            """Worker thread: wait for jobs and process results."""
            nonlocal correct_top1, correct_top5, current_count

            while True:
                try:
                    item = job_queue.get(timeout=1.0)
                except queue.Empty:
                    if worker_done.is_set() and job_queue.empty():
                        break
                    continue

                if item is None:  # Poison pill
                    break

                job_id, labels = item

                # Wait for NPU to complete
                image_feature = self.session.wait(job_id)

                # Postprocessing (CLIP-specific)
                image_feature = torch.from_numpy(self.postprocessing(image_feature))
                image_feature /= image_feature.norm(dim=-1, keepdim=True)
                logits = 100.0 * image_feature @ zeroshot_text_embedding_weight

                acc1, acc5 = self._accuracy(logits, labels, topk=(1, 5))

                with result_lock:
                    batch_size = labels.shape[0]
                    current_count += batch_size
                    correct_top1 += acc1
                    correct_top5 += acc5

                    elapsed = time.time() - wall_start_time
                    current_fps = current_count / elapsed if elapsed > 0 else 0.0

                    pbar.desc = (
                        f"ImageNet | "
                        f"Top1:{correct_top1/current_count:.2f} "
                        f"Top5:{correct_top5/current_count:.2f} "
                        f"Current_FPS:{current_fps:.1f}"
                    )
                    pbar.update(1)

                job_queue.task_done()

        # Start worker thread
        worker = threading.Thread(target=worker_thread, daemon=True)
        worker.start()

        # Main thread: submit async jobs
        for images, labels in loader:
            job_id = self.session.run_async(images, user_arg=labels)
            job_queue.put((job_id, labels))

        # Signal completion and wait for worker
        worker_done.set()
        job_queue.put(None)
        worker.join()

        pbar.close()

        total_inference_time = time.time() - wall_start_time
        return self._finalize_results(correct_top1, correct_top5, current_count, total_inference_time)

    def _finalize_results(self, correct_top1: float, correct_top5: float, total_count: int, total_time: float) -> dict:
        """Calculate and log final metrics."""
        avg_fps = total_count / total_time if total_time > 0 else 0

        print(
            f"Top1 Accuracy: {correct_top1 / total_count * 100:.2f}\n"
            f"Top5 Accuracy: {correct_top5 / total_count * 100:.2f}\n"
            f"Average FPS: {avg_fps:.2f}\n"
        )
        logger.success(
            f"@JSON <Top1 Accuracy:{correct_top1 / total_count * 100:.2f}; "
            f"Top5 Accuracy:{correct_top5 / total_count * 100:.2f}; "
            f"Average FPS:{avg_fps:.2f}>"
        )

        return {
            "performance": [correct_top1, correct_top5],
            "fps": avg_fps,
        }
