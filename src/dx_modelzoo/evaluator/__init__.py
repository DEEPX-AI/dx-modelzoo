import os
import time
from abc import ABC, abstractmethod
from collections import deque
from itertools import islice
from typing import Any, Callable

from loguru import logger
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from tqdm import tqdm

from dx_modelzoo.dataset import DatasetBase
from dx_modelzoo.enums import SessionType
from dx_modelzoo.session import SessionBase


class EvaluatorBase(ABC):
    """Evaluator Base Class with async support.

    Subclasses should implement:
    - process_batch_result(): Process single batch output and update metrics
    - compute_final_metrics(): Compute final metrics from accumulated results
    - format_progress_desc(): Format progress bar description
    """

    def __init__(
        self,
        session: SessionBase,
        dataset: DatasetBase,
        workers: int = 4,
    ):
        self.session = session
        self.dataset = dataset
        self._postprocessing = None
        self.async_queue_size = workers * session.device_count
        self.workers = min(workers, os.cpu_count() or 1)
        self._use_async = session.type == SessionType.dxruntime

        # Timing metrics
        self.total_inference_time = 0.0
        self.recent_inference_times = deque(maxlen=200)

    @property
    def postprocessing(self) -> Callable:
        if self._postprocessing is None:
            raise ValueError("Evaluator's Post Processing is not set.")
        return self._postprocessing

    def set_preprocessing(self, preprocessings: Compose) -> None:
        """Set preprocessing transforms."""
        self.dataset.preprocessing = preprocessings

    def set_postprocessing(self, postprocessing: Callable) -> None:
        """Set postprocessing function."""
        self._postprocessing = postprocessing

    def make_loader(self) -> DataLoader:
        """Create DataLoader from dataset."""
        return DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.workers,
        )

    def eval(self) -> dict:
        """Main evaluation entry point."""
        if self._use_async:
            logger.info(f"Using async evaluation with queue_size={self.async_queue_size} | CPU workers={self.workers}")
            return self._eval_async()
        return self._eval_sync()

    def _eval_sync(self) -> dict:
        """Synchronous evaluation."""
        loader = self.make_loader()
        total_len = len(loader)

        # Initialize metrics (subclass-specific)
        metrics_state = self.init_metrics()

        pbar = tqdm(enumerate(loader), total=total_len)
        for batch_idx, batch_data in pbar:
            start_time = time.time()

            # Run inference
            inputs = self.extract_inputs(batch_data)
            output = self.session.run(inputs)

            inference_time = time.time() - start_time
            self.recent_inference_times.append(inference_time)
            self.total_inference_time += inference_time

            # Process result and update metrics
            if not getattr(self, "lazy_postprocessing", False):
                output = self.postprocessing(output)
            metrics_state = self.process_batch_result(batch_data, output, metrics_state)

            # Update progress bar
            if len(self.recent_inference_times) > 0:
                current_fps = len(self.recent_inference_times) / sum(self.recent_inference_times)
            else:
                current_fps = 0.0

            pbar.desc = self.format_progress_desc(metrics_state, current_fps)

        return self.compute_final_metrics(metrics_state)

    def _eval_async(self) -> dict:
        """Asynchronous evaluation using run_async/wait pattern."""
        loader = self.make_loader()
        total_len = len(loader)

        # Initialize metrics (subclass-specific)
        metrics_state = self.init_metrics()

        wall_start_time = time.time()
        pbar = tqdm(total=total_len)

        loader_iter = iter(loader)

        while True:
            # Fetch batch of items from loader
            batch_items = list(islice(loader_iter, self.async_queue_size))
            if not batch_items:
                break

            # Submit all jobs in this batch
            job_buffer = []
            for batch_data in batch_items:
                inputs = self.extract_inputs(batch_data)
                job_id = self.session.run_async(inputs)
                job_buffer.append((job_id, batch_data))

            # Wait and process all jobs in buffer
            for job_id, batch_data in job_buffer:
                # Wait for this job to complete
                output = self.session.wait(job_id)

                # Process result and update metrics
                if not getattr(self, "lazy_postprocessing", False):
                    output = self.postprocessing(output)
                metrics_state = self.process_batch_result(batch_data, output, metrics_state)

                # Update progress
                elapsed = time.time() - wall_start_time
                current_fps = pbar.n / elapsed if elapsed > 0 else 0.0

                pbar.desc = self.format_progress_desc(metrics_state, current_fps)
                pbar.update(1)

        pbar.close()

        self.total_inference_time = time.time() - wall_start_time
        return self.compute_final_metrics(metrics_state)

    # Abstract methods that subclasses must implement

    @abstractmethod
    def init_metrics(self) -> Any:
        """Initialize metrics state.

        Returns:
            Initial metrics state (e.g., dict, list, object)
        """
        pass

    @abstractmethod
    def extract_inputs(self, batch_data: Any) -> Any:
        """Extract model inputs from batch data.

        Args:
            batch_data: Raw batch data from DataLoader

        Returns:
            Model inputs (e.g., image tensor)
        """
        pass

    @abstractmethod
    def process_batch_result(self, batch_data: Any, output: Any, metrics_state: Any) -> Any:
        """Process single batch result and update metrics.

        Args:
            batch_data: Raw batch data from DataLoader
            output: Model output (after postprocessing)
            metrics_state: Current metrics state

        Returns:
            Updated metrics state
        """
        pass

    @abstractmethod
    def compute_final_metrics(self, metrics_state: Any) -> dict:
        """Compute final metrics from accumulated state.

        Args:
            metrics_state: Final metrics state

        Returns:
            dict with 'performance' and 'fps' keys
        """
        pass

    @abstractmethod
    def format_progress_desc(self, metrics_state: Any, current_fps: float) -> str:
        """Format progress bar description.

        Args:
            metrics_state: Current metrics state
            current_fps: Current FPS

        Returns:
            Progress description string
        """
        pass


__all__ = ["EvaluatorBase"]
