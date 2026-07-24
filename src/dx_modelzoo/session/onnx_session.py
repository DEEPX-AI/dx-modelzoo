from __future__ import annotations

import ctypes
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
from loguru import logger

from dx_modelzoo.session import SessionBase
from dx_modelzoo.session.runtime_config import OnnxRuntimeConfig, RuntimeConfig


def _preload_nvidia_libs() -> None:
    """Preload pip-installed NVIDIA shared libraries for onnxruntime CUDA support.

    When nvidia-cublas-cu12 / nvidia-cudnn-cu12 etc. are pip-installed,
    their .so files live inside site-packages and the dynamic linker
    can't find them.  Loading them via ctypes before onnxruntime
    import makes them available to dlopen without LD_LIBRARY_PATH.
    """
    _NVIDIA_MODULES = [
        "nvidia.cuda_runtime.lib",
        "nvidia.cublas.lib",
        "nvidia.cufft.lib",
        "nvidia.curand.lib",
        "nvidia.cusolver.lib",
        "nvidia.cusparse.lib",
        "nvidia.cudnn.lib",
        "nvidia.nccl.lib",
        "nvidia.nvjitlink.lib",
    ]
    import importlib

    for mod_name in _NVIDIA_MODULES:
        try:
            mod = importlib.import_module(mod_name)
            for so in sorted(Path(mod.__path__[0]).glob("*.so*")):
                try:
                    ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
        except ImportError:
            continue


_preload_nvidia_libs()


def _get_ort_provider(device: Union[Literal["gpu", "cpu"], List[int], int] | None = None) -> list:
    """Detect available onnxruntime providers with graceful fallback.

    Args:
        device: "gpu" or "cpu". If specified with CUDA available,
                returns provider with device_id option set.
    """
    try:
        import onnxruntime as ort

        if device == "gpu":
            provider = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif (isinstance(device, list) and len(device) > 0) or isinstance(device, int):
            provider = [
                ("CUDAExecutionProvider", {"device_id": device}),
                "CPUExecutionProvider",
            ]
        else:
            provider = ["CPUExecutionProvider"]

        available = ort.get_available_providers()
        if "CUDAExecutionProvider" not in available:
            provider = ["CPUExecutionProvider"]

        return provider
    except Exception:
        pass
    return ["CPUExecutionProvider"]


def _get_ort_session_options():
    """Return default onnxruntime session options."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    return opts


class OnnxRuntimeSession(SessionBase):
    """ONNX Runtime inference session (pure numpy, no torch)."""

    def __init__(self, path: str, runtime_config: Optional[RuntimeConfig] = None) -> None:
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            raise ImportError(
                "onnxruntime is required for OnnxRuntimeSession. " "Install it with: pip install onnxruntime"
            )

        cfg = runtime_config if runtime_config is not None else OnnxRuntimeConfig()
        super().__init__(path)
        from onnxruntime import InferenceSession

        provider = _get_ort_provider(cfg.device)
        opts = _get_ort_session_options()
        self.inference_session = InferenceSession(self.path, opts, providers=provider)

        # Query input dtype from ONNX model metadata
        onnx_input = self.inference_session.get_inputs()[0]
        _onnx_type_map = {
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
            "tensor(double)": np.float64,
            "tensor(int32)": np.int32,
            "tensor(int64)": np.int64,
            "tensor(int8)": np.int8,
            "tensor(uint8)": np.uint8,
        }
        self._input_dtype = _onnx_type_map.get(onnx_input.type, np.float32)

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
        self._futures: Dict[int, Any] = {}
        self._results: Dict[int, Union[List[np.ndarray], Exception]] = {}
        self._results_lock = threading.Lock()
        # ONNX defaults to sync eval (no inference_engine callback); honor an
        # explicit runtime async flag when provided (OnnxRuntimeConfig.ASYNC_DEFAULT
        # == False).
        self._use_async = cfg.use_async

    @property
    def dtype(self):
        return [self._input_dtype]

    def run(self, inputs: Union[np.ndarray, Dict[str, np.ndarray]]) -> List[np.ndarray]:
        # Multi-input: dict mapping input name -> np.ndarray
        if isinstance(inputs, dict):
            input_specs = self.inference_session.get_inputs()
            type_map = {
                "tensor(float)": np.float32,
                "tensor(float16)": np.float16,
                "tensor(double)": np.float64,
                "tensor(int64)": np.int64,
                "tensor(int32)": np.int32,
                "tensor(uint8)": np.uint8,
            }
            feed = {}
            for spec in input_specs:
                if spec.name not in inputs:
                    continue
                arr = np.asarray(inputs[spec.name])
                feed[spec.name] = arr.astype(type_map.get(spec.type, self._input_dtype))
            missing = [spec.name for spec in input_specs if spec.name not in feed]
            if missing:
                raise KeyError(f"Missing input tensor(s): {missing}. Provided: {list(inputs.keys())}")
            return self.inference_session.run([], feed)

        if not isinstance(inputs, np.ndarray):
            inputs = np.asarray(inputs)
        name = self.inference_session.get_inputs()[0].name
        return self.inference_session.run([], {name: inputs.astype(self._input_dtype)})

    def run_async(self, inputs: Union[np.ndarray, Dict[str, np.ndarray]], **kwargs) -> int:
        if isinstance(inputs, dict):
            # For dict inputs, run synchronously and store result for wait()
            result = self.run(inputs)
            with self._job_counter_lock:
                self._job_counter += 1
                job_id = self._job_counter
            with self._results_lock:
                self._results[job_id] = result
            return job_id

        if not isinstance(inputs, np.ndarray):
            inputs = np.asarray(inputs)
        output_buffer = kwargs.get("output_buffer")

        def _infer(inp, out_buf):
            name = self.inference_session.get_inputs()[0].name
            out = self.inference_session.run([], {name: inp.astype(self._input_dtype)})
            if out_buf is not None:
                try:
                    if isinstance(out_buf, list):
                        for i, arr in enumerate(out):
                            if i < len(out_buf) and out_buf[i] is not None:
                                np.copyto(out_buf[i], arr)
                    else:
                        np.copyto(out_buf, out[0])
                except Exception as e:
                    logger.debug("copyto output_buffer failed: {}", e)
            return out

        with self._job_counter_lock:
            self._job_counter += 1
            job_id = self._job_counter
            future = self._executor.submit(_infer, inputs, output_buffer)
            self._futures[job_id] = future

        future.add_done_callback(lambda fut, jid=job_id: self._on_done(jid, fut))

        return job_id

    def wait(self, job_id: int) -> List[np.ndarray]:
        with self._results_lock:
            if job_id in self._results:
                res = self._results.pop(job_id)
                if isinstance(res, Exception):
                    raise res
                return res

        with self._job_counter_lock:
            future = self._futures.get(job_id)
        if future is None:
            # Re-check _results: _on_done may have stored result and removed future
            with self._results_lock:
                if job_id in self._results:
                    res = self._results.pop(job_id)
                    if isinstance(res, Exception):
                        raise res
                    return res
            raise ValueError(f"Unknown job_id: {job_id}")

        result = future.result()
        with self._job_counter_lock:
            self._futures.pop(job_id, None)
        with self._results_lock:
            self._results.pop(job_id, None)
        return result

    def _on_done(self, job_id: int, future) -> None:
        try:
            res = future.result()
        except Exception as e:
            res = e

        with self._results_lock:
            self._results[job_id] = res
        with self._job_counter_lock:
            self._futures.pop(job_id, None)

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass
        with self._job_counter_lock:
            self._futures.clear()
        with self._results_lock:
            self._results.clear()
