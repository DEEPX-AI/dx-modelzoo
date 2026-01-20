import os
from typing import Any, List, Optional, Union

import numpy as np
import torch
from dx_engine import InferenceEngine, InferenceOption

from dx_modelzoo.enums import SessionType
from dx_modelzoo.session import SessionBase
from dx_modelzoo.utils import torch_to_numpy


class DxRuntimeSession(SessionBase):
    def __init__(self, path: str) -> None:
        super().__init__(path, SessionType.dxruntime)
        io = InferenceOption()
        io.devices = self._get_devices_from_env()
        self.inference_engine = InferenceEngine(self.path, io)
        self.dtype = self.inference_engine.get_input_data_type()

        # Keep references to pending async inputs to prevent GC
        self._pending_inputs: dict[int, np.ndarray] = {}

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
            return []
        return [int(d.strip()) for d in env_devices.split(",") if d.strip()]

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
        # Keep reference to prevent GC while NPU is processing
        self._pending_inputs[job_id] = inputs
        return job_id

    def wait(self, job_id: int) -> List[np.ndarray]:
        """Wait for an asynchronous job to complete and retrieve its output.

        Args:
            job_id: The job ID returned from run_async.

        Returns:
            List of output arrays.
        """
        result = self.inference_engine.wait(job_id)
        # Release the input reference now that processing is complete
        self._pending_inputs.pop(job_id, None)
        return result
