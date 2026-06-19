from __future__ import annotations

import os
from typing import Dict, List, Optional, Union

import numpy as np

from dx_modelzoo.session import SessionBase
from dx_modelzoo.session.runtime_config import DxnnRuntimeConfig, RuntimeConfig


class DxRuntimeSession(SessionBase):
    """DxRuntime inference session (requires dx_engine)."""

    def __init__(self, path: str, runtime_config: Optional[RuntimeConfig] = None) -> None:
        try:
            from dx_engine import DeviceStatus, InferenceEngine, InferenceOption
        except ImportError:
            raise ImportError(
                "dx_engine is required for DxRuntimeSession. " "Please install the DeepX runtime package."
            )

        cfg = runtime_config if runtime_config is not None else DxnnRuntimeConfig()
        super().__init__(path)
        # DXNN supports async inference; honor an explicit runtime async flag,
        # otherwise default to async.
        self._use_async = cfg.use_async
        os.environ["DXRT_DYNAMIC_CPU_THREAD"] = "ON"
        io = InferenceOption()
        # ``buffer_count``/``use_ort`` are DXNN-only fields (DxnnRuntimeConfig).
        # Leave the engine default untouched when buffer_count is unset (None);
        # tolerate a base RuntimeConfig that lacks these fields.
        buffer_count = getattr(cfg, "buffer_count", None)
        if buffer_count is not None:
            io.buffer_count = buffer_count
        io.use_ort = getattr(cfg, "use_ort", True)
        io.devices = self._get_devices_from_env(DeviceStatus, cfg.device)
        self.inference_engine = InferenceEngine(self.path, io)
        self.dtype = self.inference_engine.get_input_data_type()
        self._inflight_inputs: Dict = {}
        self._closed = False

    def _get_devices_from_env(self, DeviceStatus, devices: Union[int, str, List[int], List[str]] = None) -> List[int]:
        if devices is None:
            devices = []
        else:
            if isinstance(devices, int):
                devices = [devices]
            if isinstance(devices, str):
                devices = [int(d.strip()) for d in devices.split(",") if d.strip()]
            elif isinstance(devices, list):
                devices = [int(d) for d in devices]
            else:
                raise ValueError("Device must be an integer, a string, or a list of integers.")

        env_devices = os.environ.get("DXNN_DEVICES", "")
        if env_devices.strip():
            devices = [int(d.strip()) for d in env_devices.split(",") if d.strip()]

        if len(devices):
            self.device_count = len(devices)
        else:
            self.device_count = DeviceStatus.get_device_count()
        return devices

    def run(self, inputs: Union[np.ndarray, Dict[str, np.ndarray]]) -> List[np.ndarray]:
        if isinstance(inputs, dict):
            if len(inputs) == 1:
                inputs = next(iter(inputs.values()))
            else:
                names = self.inference_engine.get_input_tensor_names()
                dtypes = self.inference_engine.get_input_data_type()
                prepared = {}
                for name, dt in zip(names, dtypes):
                    if name not in inputs:
                        raise KeyError(f"Missing input tensor '{name}'. Provided: {list(inputs.keys())}")
                    arr = inputs[name]
                    if not isinstance(arr, np.ndarray):
                        arr = np.asarray(arr)
                    if arr.dtype != dt:
                        arr = arr.astype(dt)
                    if not arr.flags["C_CONTIGUOUS"]:
                        arr = np.ascontiguousarray(arr)
                    prepared[name] = arr
                return self.inference_engine.run_multi_input(prepared)

        if not isinstance(inputs, np.ndarray):
            inputs = np.asarray(inputs)
        if self.dtype[0] != inputs.dtype:
            inputs = inputs.astype(self.dtype[0])
        if not inputs.flags["C_CONTIGUOUS"]:
            inputs = np.ascontiguousarray(inputs)
        return self.inference_engine.Run([inputs])

    def run_async(self, inputs: Union[np.ndarray, Dict[str, np.ndarray]], **kwargs) -> int:
        user_arg = kwargs.get("user_arg")
        output_buffer = kwargs.get("output_buffer")
        if isinstance(inputs, dict):
            if len(inputs) == 1:
                inputs = next(iter(inputs.values()))
            else:
                names = self.inference_engine.get_input_tensor_names()
                dtypes = self.inference_engine.get_input_data_type()
                prepared = {}
                for name, dt in zip(names, dtypes):
                    if name not in inputs:
                        raise KeyError(f"Missing input tensor '{name}'. Provided: {list(inputs.keys())}")
                    arr = inputs[name]
                    if not isinstance(arr, np.ndarray):
                        arr = np.asarray(arr)
                    if arr.dtype != dt:
                        arr = arr.astype(dt)
                    if not arr.flags["C_CONTIGUOUS"]:
                        arr = np.ascontiguousarray(arr)
                    prepared[name] = arr
                job_id = self.inference_engine.run_async_multi_input(prepared, user_arg, output_buffer)
                self._inflight_inputs[job_id] = prepared
                return job_id

        if not isinstance(inputs, np.ndarray):
            inputs = np.asarray(inputs)
        if self.dtype[0] != inputs.dtype:
            inputs = inputs.astype(self.dtype[0])
        if not inputs.flags["C_CONTIGUOUS"]:
            inputs = np.ascontiguousarray(inputs)
        job_id = self.inference_engine.run_async(inputs, user_arg, output_buffer)
        # Store reference so the buffer stays alive until C++ finishes reading it.
        # Without this, dtype/contiguous conversions above create a temporary
        # array that gets freed before the async engine is done with it.
        self._inflight_inputs[job_id] = inputs
        return job_id

    def wait(self, job_id: int) -> List[np.ndarray]:
        result = self.inference_engine.wait(job_id)
        self._inflight_inputs.pop(job_id, None)
        return result

    def get_output_specs(self) -> List[dict]:
        return self.inference_engine.get_output_tensors_info()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            del self.inference_engine
        except Exception:
            pass
