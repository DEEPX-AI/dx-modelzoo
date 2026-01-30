import os
from typing import Any, List, Optional, Union

import numpy as np
import torch
from dx_engine import DeviceStatus, InferenceEngine, InferenceOption

from dx_modelzoo.enums import SessionType
from dx_modelzoo.session import SessionBase
from dx_modelzoo.utils import torch_to_numpy


class DxRuntimeSession(SessionBase):
    def __init__(self, path: str) -> None:
        super().__init__(path, SessionType.dxruntime)
        self.device_count = 1
        io = InferenceOption()
        io.use_ort = True
        io.devices = self._get_devices_from_env()
        self.inference_engine = InferenceEngine(self.path, io)
        self.dtype = self.inference_engine.get_input_data_type()

        self._closed = False

    def _get_devices_from_env(self) -> List[int]:
        """Get device list from DXNN_DEVICES environment variable.

        Examples:
            DXNN_DEVICES="0" -> [0]
            DXNN_DEVICES="0,1,2" -> [0, 1, 2]
            DXNN_DEVICES="" or not set -> []

        Returns:
            List of device IDs.
        """
        env_devices = os.environ.get("DXNN_DEVICES", "")
        if not env_devices.strip():
            devices = []
        else:
            devices = [int(d.strip()) for d in env_devices.split(",") if d.strip()]

        if len(devices):
            self.device_count = len(devices)
        else:
            self.device_count = DeviceStatus.get_device_count()
    
        return devices

    def run(self, inputs: torch.Tensor) -> List[np.ndarray]:
        inputs = torch_to_numpy(inputs)

        if self.dtype[0] != inputs.dtype:
            inputs = inputs.astype(self.dtype[0])

        return self.inference_engine.Run([inputs])

    def run_async(
        self,
        inputs: torch.Tensor,
        user_arg: Any = None,
        output_buffer: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    ) -> int:
        """Run inference asynchronously.

        Args:
            inputs: Input tensor.
            user_arg: Optional user-defined argument to be passed to the callback.
            output_buffer: Optional pre-allocated buffer for the output.

        Returns:
            job_id for this asynchronous operation.
        """
        inputs = torch_to_numpy(inputs)

        if self.dtype[0] != inputs.dtype:
            inputs = inputs.astype(self.dtype[0])

        job_id = self.inference_engine.run_async(inputs, user_arg, output_buffer)
        return job_id

    def wait(self, job_id: int) -> List[np.ndarray]:
        """Wait for an asynchronous job to complete and retrieve its output.

        Args:
            job_id: The job ID returned from run_async.

        Returns:
            List of output arrays.
        """
        result = self.inference_engine.wait(job_id)
        return result

    def close(self) -> None:
        """Clean up resources and pending async jobs.

        This method should be called when the session is no longer needed
        to prevent memory leaks from pending async operations.
        """
        if self._closed:
            return

        self._closed = True

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures resources are cleaned up."""
        self.close()
        return False

    def __del__(self):
        """Destructor - final cleanup attempt."""
        try:
            if not self._closed:
                self.close()
        except Exception:
            # Suppress exceptions in destructor
            pass
