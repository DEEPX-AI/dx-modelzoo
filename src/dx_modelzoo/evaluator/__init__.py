from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from loguru import logger
from tqdm import tqdm

from dx_modelzoo.common.dataloader import DataLoader, DatasetBase
from dx_modelzoo.common.registry import Registry
from dx_modelzoo.session import SessionBase

__all__ = ["EVALUATOR_REGISTRY", "EvaluatorBase"]

EVALUATOR_REGISTRY = Registry("evaluator")


class EvaluatorBase(ABC):
    """Base evaluator with sync/async eval support."""

    def __init__(
        self,
        session: SessionBase,
        dataset: DatasetBase,
        workers: int = 12,
        batch_size: int = 1,
    ) -> None:
        self.session = session
        self.dataset = dataset
        self._postprocessing = None
        self.model_spec = {}
        self.batch_size = batch_size
        self.workers = min(workers, os.cpu_count() or 1)
        self.async_queue_size = workers * getattr(session, "device_count", 1) * 2
        # The session owns the async decision (derived from its runtime config
        # or a sensible default). The evaluator simply honors it.
        self._use_async = getattr(session, "_use_async", False)

        self.total_inference_time = 0.0
        self.recent_inference_times: deque = deque(maxlen=200)

        # Evaluation context — set by caller before eval()
        self.model_name: str = ""
        self.display_name: Optional[str] = None
        self.dataset_name: str = ""
        self.profile_name: str = ""
        self.task_name: str = getattr(self.__class__, "__registry_key__", "")
        self._start_time: Optional[datetime] = None

    @property
    def postprocessing(self) -> Callable:
        if self._postprocessing is None:
            raise ValueError("Evaluator's postprocessing is not set.")
        return self._postprocessing

    def set_preprocessing(self, preprocessing) -> None:
        self.dataset.preprocessing = preprocessing

    def set_postprocessing(self, postprocessing) -> None:
        self._postprocessing = postprocessing

    def _build_postprocessing_context(self, batch_data) -> dict:
        """Return kwargs to forward to the postprocessing pipeline.

        Override in subclasses that need to pass extra context (e.g.
        ``origin_hw``, ``input_hw``) to the postprocessor.
        """
        return {}

    def _get_runtime_target(self) -> Optional[str]:
        suffix = Path(getattr(self.session, "path", "")).suffix.lower()
        if suffix == ".onnx":
            return "onnx"
        if suffix == ".dxnn":
            return "dxnn"
        return None

    def _run_postprocessing(self, output, batch_data):
        """Run postprocessing on model output.

        Override in subclasses that need custom handling (e.g. splitting
        multi-head outputs before postprocessing).
        """
        kwargs = self._build_postprocessing_context(batch_data)
        runtime_target = self._get_runtime_target()
        if runtime_target is not None:
            kwargs.setdefault("runtime_target", runtime_target)
        return self.postprocessing(output, **kwargs)

    def make_loader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            prefetch_factor=2,
        )

    def eval(self) -> dict:
        self._start_time = datetime.now()
        try:
            if self._use_async:
                return self._eval_async()
            return self._eval_sync()
        except Exception as e:
            elapsed = (datetime.now() - self._start_time).total_seconds()
            logger.error(f"@JSON {self.model_name} <Error during evaluation: {str(e)}>")

            import traceback

            traceback_str = traceback.format_exc()
            logger.debug(f"Stack trace:\n{traceback_str}")

            return self._build_result(
                metrics=[],
                fps=0.0,
                elapsed_time=elapsed,
                error=str(e),
            )

    def _eval_sync(self) -> dict:
        loader = self.make_loader()
        total_len = len(loader)
        if total_len == 0:
            raise ValueError("Dataset is empty. Cannot perform evaluation.")
        metrics_state = self.init_metrics()

        total_load_time = 0.0
        total_post_time = 0.0

        logger.info(f"Starting sync eval — {total_len} batches")

        loader_iter = iter(loader)
        first_batch = next(loader_iter)

        pbar = tqdm(enumerate(chain([first_batch], loader_iter)), total=total_len)
        iter_start = time.time()
        for batch_idx, batch_data in pbar:
            load_time = time.time() - iter_start
            total_load_time += load_time

            infer_start = time.time()
            inputs = self.extract_inputs(batch_data)
            output = self.session.run(inputs)
            inference_time = time.time() - infer_start

            self.recent_inference_times.append(inference_time)
            self.total_inference_time += inference_time

            post_start = time.time()
            output = self._run_postprocessing(output, batch_data)
            metrics_state = self.process_batch_result(batch_data, output, metrics_state)
            total_post_time += time.time() - post_start

            if len(self.recent_inference_times) > 0:
                current_fps = len(self.recent_inference_times) / sum(self.recent_inference_times)
            else:
                current_fps = 0.0

            pbar.desc = self.format_progress_desc(metrics_state, current_fps)
            iter_start = time.time()

        wall_total = total_load_time + self.total_inference_time + total_post_time
        if wall_total > 0:
            logger.info(
                f"Time breakdown — "
                f"load: {total_load_time:.1f}s ({total_load_time / wall_total * 100:.0f}%) | "
                f"infer: {self.total_inference_time:.1f}s ({self.total_inference_time / wall_total * 100:.0f}%) | "
                f"post: {total_post_time:.1f}s ({total_post_time / wall_total * 100:.0f}%)"
            )

        return self.compute_final_metrics(metrics_state)

    def _eval_async(self) -> dict:
        """Async pipeline: loader pool prefetches, main thread drives inference.

        Pipeline:
            LoaderPool(prefetch) → ready_queue → Main(run_async → wait → postprocess → metrics)

        No callbacks, no inter-thread queues beyond data loading.
        Hang-free: stop_event signals loader to exit, queue drain prevents deadlock.
        """
        import queue
        import threading
        from collections import deque
        from concurrent.futures import ThreadPoolExecutor

        total = len(self.dataset)
        if total == 0:
            raise ValueError("Dataset is empty. Cannot perform evaluation.")
        metrics_state = self.init_metrics()
        device_count = getattr(self.session, "device_count", 1)
        max_inflight = min(device_count * 4, total)

        logger.info(
            f"Starting async eval — {total} samples, "
            f"{self.workers} loader threads, {device_count} devices, "
            f"max_inflight={max_inflight}"
        )

        # --- Loader thread: prefetch dataset items into ready_queue ---
        ready_queue: queue.Queue = queue.Queue(maxsize=self.workers * 2)
        stop_event = threading.Event()
        loader_error: list = []

        def _loader_worker():
            try:
                with ThreadPoolExecutor(max_workers=self.workers) as pool:
                    futures = deque()
                    submitted = 0
                    prefetch = min(self.workers * 4, total)
                    while submitted < prefetch:
                        idx = submitted
                        futures.append((idx, pool.submit(self.dataset.__getitem__, idx)))
                        submitted += 1
                    order = 0
                    while futures and not stop_event.is_set():
                        idx, fut = futures.popleft()
                        data = fut.result()
                        while not stop_event.is_set():
                            try:
                                ready_queue.put((order, data), timeout=1)
                                break
                            except queue.Full:
                                continue
                        if stop_event.is_set():
                            break
                        order += 1
                        if submitted < total:
                            next_idx = submitted
                            futures.append(
                                (
                                    next_idx,
                                    pool.submit(self.dataset.__getitem__, next_idx),
                                )
                            )
                            submitted += 1
            except Exception as e:
                loader_error.append(e)
            finally:
                ready_queue.put(None)

        loader_thread = threading.Thread(target=_loader_worker, daemon=True, name="loader")
        loader_thread.start()

        # --- Main thread: bounded async inference + postprocess ---
        wall_start = time.time()
        pbar = tqdm(total=total)
        inflight: deque = deque()  # (job_id, idx, batch_data)
        processed = 0

        try:
            items_done = False
            while processed < total:
                # Fill inflight up to max
                while len(inflight) < max_inflight and not items_done:
                    item = ready_queue.get()
                    if item is None:
                        items_done = True
                        break
                    idx, batch_data = item
                    inputs = self.extract_inputs(batch_data)
                    job_id = self.session.run_async(inputs)
                    inflight.append((job_id, idx, batch_data))

                if not inflight:
                    break

                # Wait for oldest job
                job_id, idx, batch_data = inflight.popleft()
                output = self.session.wait(job_id)

                # Postprocess + metrics
                output = self._run_postprocessing(output, batch_data)
                metrics_state = self.process_batch_result(batch_data, output, metrics_state)
                processed += 1

                elapsed = time.time() - wall_start
                fps = processed / elapsed if elapsed > 0 else 0
                pbar.desc = self.format_progress_desc(metrics_state, fps)
                pbar.update(1)
        finally:
            pbar.close()
            stop_event.set()
            # Drain queue to unblock any in-progress put()
            while not ready_queue.empty():
                try:
                    ready_queue.get_nowait()
                except queue.Empty:
                    break
            loader_thread.join(timeout=5)

        if loader_error:
            raise loader_error[0]

        self.total_inference_time = time.time() - wall_start
        if self.total_inference_time > 0:
            throughput = total / self.total_inference_time
            logger.info(
                f"Async eval — wall: {self.total_inference_time:.1f}s | " f"throughput: {throughput:.1f} samples/s"
            )
        return self.compute_final_metrics(metrics_state)

    def _build_result(
        self,
        metrics: List[Dict[str, Any]],
        fps: float,
        elapsed_time: float,
        error: Optional[str] = None,
    ) -> dict:
        """Build standardized result dict."""
        result = {
            "model": self.model_name,
            "operations": self.model_spec.get("operations", None),
            "parameters": self.model_spec.get("parameters", None),
            "license": self.model_spec.get("license", None),
            "task": self.task_name,
            "input_resolution": self.model_spec.get("input_resolution", None),
            "dataset": self.dataset_name,
            "metrics": metrics,
            "fps": round(float(fps), 2),
            "elapsed_time": int(round(elapsed_time)),
            "start_time": self._start_time.strftime("%Y-%m-%d %H:%M:%S") if self._start_time else "",
            "profile": self.profile_name,
        }
        if getattr(self, "display_name", None):
            result["display_name"] = self.display_name
        if error is not None:
            result["error"] = error
        return result

    def _finalize(
        self,
        metric_names: List[str],
        metric_values: List[float],
        fps: float,
    ) -> dict:
        """Common finalize: log success message + build result dict."""
        elapsed = (datetime.now() - self._start_time).total_seconds() if self._start_time else self.total_inference_time
        metrics = [{"name": n, "metric_value": round(float(v), 4)} for n, v in zip(metric_names, metric_values)]

        # Build log message
        parts = "; ".join(f"{n}:{v:.6f}" for n, v in zip(metric_names, metric_values))
        logger.success(f"@JSON {self.model_name} <{parts}; Average FPS:{fps:.2f}>")

        return self._build_result(
            metrics=metrics,
            fps=fps,
            elapsed_time=elapsed,
        )

    @abstractmethod
    def init_metrics(self) -> Any:
        ...

    @abstractmethod
    def extract_inputs(self, batch_data: Any) -> Any:
        ...

    @abstractmethod
    def process_batch_result(self, batch_data: Any, output: Any, metrics_state: Any) -> Any:
        ...

    @abstractmethod
    def compute_final_metrics(self, metrics_state: Any) -> dict:
        ...

    @abstractmethod
    def format_progress_desc(self, metrics_state: Any, current_fps: float) -> str:
        ...


from dx_modelzoo.evaluator import depth_estimation_evaluator  # noqa: F401, E402
from dx_modelzoo.evaluator import face_attribute_evaluator  # noqa: F401, E402
from dx_modelzoo.evaluator import face_detection_evaluator  # noqa: F401, E402
from dx_modelzoo.evaluator import face_landmark_evaluator  # noqa: F401, E402
from dx_modelzoo.evaluator import face_recognition_evaluator  # noqa: F401, E402
from dx_modelzoo.evaluator import hand_detection_evaluator  # noqa: F401, E402
from dx_modelzoo.evaluator import hand_landmark_evaluator  # noqa: F401, E402
from dx_modelzoo.evaluator import image_classification_evaluator  # noqa: F401, E402
from dx_modelzoo.evaluator import image_denoising_evaluator  # noqa: F401, E402
from dx_modelzoo.evaluator import instance_segmentation_evaluator  # noqa: F401, E402
from dx_modelzoo.evaluator import keypoint_detection_evaluator  # noqa: F401, E402
from dx_modelzoo.evaluator import low_light_enhancement_evaluator  # noqa: F401, E402
from dx_modelzoo.evaluator import object_detection_3d_evaluator  # noqa: F401, E402
from dx_modelzoo.evaluator import object_detection_evaluator  # noqa: F401, E402
from dx_modelzoo.evaluator import object_pose_estimation_evaluator  # noqa: F401, E402
from dx_modelzoo.evaluator import oriented_object_detection_evaluator  # noqa: F401, E402
from dx_modelzoo.evaluator import panoptic_driving_perception_evaluator  # noqa: F401, E402
from dx_modelzoo.evaluator import person_attribute_evaluator  # noqa: F401, E402
from dx_modelzoo.evaluator import pose_estimation_evaluator  # noqa: F401, E402
from dx_modelzoo.evaluator import pose_estimation_topdown_evaluator  # noqa: F401, E402
from dx_modelzoo.evaluator import semantic_segmentation_evaluator  # noqa: F401, E402
from dx_modelzoo.evaluator import super_resolution_evaluator  # noqa: F401, E402
from dx_modelzoo.evaluator import visual_place_recognition_evaluator  # noqa: F401, E402
from dx_modelzoo.evaluator import zero_shot_image_classification_evaluator  # noqa: F401, E402
from dx_modelzoo.evaluator import zero_shot_instance_segmentation_evaluator  # noqa: F401, E402
