from typing import List

import numpy as np
import torch
from dx_engine import InferenceEngine

from dx_modelzoo.enums import SessionType
from dx_modelzoo.session import SessionBase
from dx_modelzoo.utils import torch_to_numpy


class DxRuntimeSession(SessionBase):
    def __init__(self, path: str) -> None:
        super().__init__(path, SessionType.dxruntime)
        self.inference_engine = InferenceEngine(self.path)

    def run(self, inputs: torch.Tensor) -> List[np.ndarray]:
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        inputs = torch_to_numpy(inputs)
        return self.inference_engine.Run([inputs])
