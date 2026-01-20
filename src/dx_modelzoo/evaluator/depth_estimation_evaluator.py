import math
import queue
import threading
import time
from collections import deque

import torch
from loguru import logger
from tqdm import tqdm

from dx_modelzoo.enums import SessionType
from dx_modelzoo.evaluator import EvaluatorBase


class DepthEstimationEvaluator(EvaluatorBase):
    def __init__(self, session, dataset) -> None:
        super().__init__(session, dataset)

    def eval(self):
        # Use async evaluation for DX runtime
        if self.session.type == SessionType.dxruntime:
            return self._eval_async()

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

    def _eval_async(self):
        """Async evaluation for DX runtime using run_async/wait pattern."""
        loader = self.make_loader()
        total_len = len(loader)

        # Shared state with lock
        lock = threading.Lock()
        total_inference_time = 0.0
        recent_inference_times = deque(maxlen=50)
        rmse_sum = 0.0
        worker_error = None

        # Queue for pending jobs: (job_id, depth, submit_time)
        pending_queue = queue.Queue(maxsize=32)
        done_event = threading.Event()

        pbar = tqdm(total=total_len, desc="Depth Estimation (Async)")

        def result_worker():
            """Worker thread that collects results and calculates RMSE."""
            nonlocal total_inference_time, rmse_sum, worker_error

            while True:
                try:
                    item = pending_queue.get(timeout=0.1)
                except queue.Empty:
                    if done_event.is_set():
                        break
                    continue

                if item is None:  # Sentinel
                    break

                job_id, depth, submit_time = item

                try:
                    # Wait for NPU result
                    output = self.session.wait(job_id)
                    inference_time = time.time() - submit_time

                    # Postprocessing
                    output = self.postprocessing(output)
                    output = torch.from_numpy(output)

                    # Calculate RMSE
                    valid_mask = ((depth > 0) + (output > 0)) > 0
                    output_masked = output[valid_mask]
                    depth_masked = depth[valid_mask]
                    abs_diff = (output_masked - depth_masked).abs()
                    mse = float((torch.pow(abs_diff, 2)).mean())
                    rmse = math.sqrt(mse)

                    with lock:
                        recent_inference_times.append(inference_time)
                        total_inference_time += inference_time
                        rmse_sum += rmse

                        if len(recent_inference_times) > 0:
                            current_fps = len(recent_inference_times) / sum(recent_inference_times)
                        else:
                            current_fps = 0.0

                        pbar.set_description(f"Depth | FPS: {current_fps:.1f}")
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
            for images, depth in loader:
                with lock:
                    if worker_error is not None:
                        raise worker_error

                submit_time = time.time()
                job_id = self.session.run_async(images)

                pending_queue.put((job_id, depth, submit_time))

            # Signal completion and wait for worker
            done_event.set()
            pending_queue.put(None)  # Sentinel
            worker_thread.join()

            if worker_error is not None:
                raise worker_error

        finally:
            pbar.close()

        avg_fps = total_len / total_inference_time if total_inference_time > 0 else 0.0
        avg_rmse = rmse_sum / total_len

        print(f"RMSE: {round(avg_rmse, 3)}")
        print(f"Average FPS: {avg_fps:.2f}")
        logger.success(f"@JSON <RMSE:{round(avg_rmse, 3)}; Average FPS:{avg_fps:.2f}>")

        return {
            "performance": [avg_rmse],
            "fps": avg_fps,
        }
