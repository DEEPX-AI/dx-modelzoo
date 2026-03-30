import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Union

import numpy as np
import onnxruntime as ort
import torch
from onnx import ModelProto
from onnxruntime import InferenceSession

from dx_modelzoo.enums import SessionType
from dx_modelzoo.exceptions import InvalidPathError
from dx_modelzoo.session import SessionBase
from dx_modelzoo.utils import torch_to_numpy


def get_ort_provider() -> List[str]:
    """get onnxruntime provider.
    if cuda is available, return "CUDAExecutionProvider". else return "CPUExecutionProvider"

    Returns:
        List[str]: onnxruntime provider.
    """
    if torch.cuda.is_available():
        provider = ["CUDAExecutionProvider"]
        print(f"Found {torch.cuda.device_count()} GPU(s), using GPU. ")
    else:
        provider = ["CPUExecutionProvider"]
        print("No GPU is available, using CPU.")
    return provider
    # return ["CPUExecutionProvider"]


def get_ort_session_options() -> ort.SessionOptions:
    """get onnxruntime session options.
    it sets graph_optimization_level to ORT_ENABLE_BASIC.

    Returns:
        ort.SessionOptions: onnxruntime session options.
    """
    sess_option = ort.SessionOptions()
    sess_option.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    return sess_option


class OnnxRuntimeSession(SessionBase):
    """OnnxRuntimeSession class.

    Args:
        path (str): onnx model path.
    """

    def __init__(self, path: str):
        super().__init__(path, SessionType.onnxruntime)
        self.device_count = 1
        self.inference_session = self._get_inference_session()
        # Async support: use a thread pool to run blocking inference without blocking caller
        # max workers configurable via ONNX_ASYNC_WORKERS env var
        try:
            workers_env = os.environ.get("ONNX_ASYNC_WORKERS")
            if workers_env is not None and workers_env.strip():
                max_workers = max(1, int(workers_env))
            else:
                cores = os.cpu_count() or 4
                max_workers = max(4, cores)
        except Exception:
            max_workers = 4

        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._job_counter = 0
        self._job_counter_lock = threading.Lock()
        self._futures = {}  # job_id -> Future

        # Completed results cache: job_id -> result or Exception
        self._results: Dict[int, Union[List[np.ndarray], Exception]] = {}
        try:
            self._results_lock = threading.Lock()
        except Exception:
            self._results_lock = None

    def _get_inference_session(self) -> InferenceSession:
        """get onnxruntime inference session.

        Returns:
            InferenceSession: onnxruntime inference session.
        """
        provider = get_ort_provider()
        sess_option = get_ort_session_options()

        return InferenceSession(self.path, sess_option, providers=provider)

    def run(self, inputs: torch.Tensor) -> List[np.ndarray]:
        """run onnxruntime session.

        Args:
            inputs (np.ndarray): inputs.

        Returns:
            List[np.ndarray]: outputs.
        """
        inputs = torch_to_numpy(inputs)
        sess_inputs_name = self.inference_session.get_inputs()[0].name
        sess_inputs = {sess_inputs_name: inputs.astype(np.float32)}
        return self.inference_session.run([], sess_inputs)

    def run_async(
        self,
        inputs: torch.Tensor,
        user_arg: Any = None,
        output_buffer: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    ) -> int:
        """Run inference asynchronously. Returns a job id."""
        inputs = torch_to_numpy(inputs)

        def _call_inference(inp, out_buf):
            sess_inputs_name = self.inference_session.get_inputs()[0].name
            sess_inputs = {sess_inputs_name: inp.astype(np.float32)}
            out = self.inference_session.run([], sess_inputs)
            # Optionally copy to provided output buffer(s)
            if out_buf is not None:
                try:
                    if isinstance(out_buf, list):
                        for i, arr in enumerate(out):
                            if i < len(out_buf) and out_buf[i] is not None:
                                np.copyto(out_buf[i], arr)
                    else:
                        np.copyto(out_buf, out[0])
                except Exception:
                    # If copying fails, ignore and return original output
                    pass
            return out

        # Reserve job id and store pending input before scheduling to avoid GC window
        with self._job_counter_lock:
            self._job_counter += 1
            job_id = self._job_counter

        future = self._executor.submit(_call_inference, inputs, output_buffer)

        # attach done callback to free pending inputs and cache result
        try:
            future.add_done_callback(lambda fut, jid=job_id: self._on_future_done(jid, fut))
        except Exception:
            # older Python or unexpected failure: best-effort
            pass

        with self._job_counter_lock:
            self._futures[job_id] = future

        return job_id

    def wait(self, job_id: int, timeout: Optional[float] = None) -> List[np.ndarray]:
        """Wait for async job to complete and return outputs.

        This method first checks the completed results cache populated by the
        done-callback so callers can retrieve results even if the job already
        completed and the Future was cleared.
        """
        # Check cached results first
        if self._results_lock is not None:
            with self._results_lock:
                if job_id in self._results:
                    res = self._results.pop(job_id)
                    if isinstance(res, Exception):
                        raise res
                    return res
        else:
            if job_id in self._results:
                res = self._results.pop(job_id)
                if isinstance(res, Exception):
                    raise res
                return res

        # Fallback to waiting on future
        future = self._futures.get(job_id)
        if future is None:
            raise ValueError(f"Unknown job_id: {job_id}")

        result = future.result(timeout=timeout)

        # Clean up stored future if still present
        try:
            del self._futures[job_id]
        except KeyError:
            pass

        return result

    def _on_future_done(self, job_id: int, future) -> None:
        """Callback when futures complete: cache result/exception and free inputs."""
        try:
            res = future.result()
        except Exception as e:
            res = e

        # Cache result (or exception)
        if self._results_lock is not None:
            with self._results_lock:
                self._results[job_id] = res
        else:
            self._results[job_id] = res

        # Remove future reference to reduce memory
        try:
            with self._job_counter_lock:
                self._futures.pop(job_id, None)
        except Exception:
            # best-effort cleanup
            self._futures.pop(job_id, None)

    def close(self, wait: bool = False) -> None:
        """Shutdown the executor and clear internal caches.

        Args:
            wait: if True, wait for currently running jobs to finish.
        """
        try:
            self._executor.shutdown(wait=wait)
        except Exception:
            pass

        # Clear stored references
        self._futures.clear()
        self._results.clear()
