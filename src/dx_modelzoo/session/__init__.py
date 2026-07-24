from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np

__all__ = ["SessionBase"]


class SessionBase(ABC):
    def __init__(self, path: str, device_count: int = 1) -> None:
        self.path = path
        self.device_count = device_count
        # Sessions own the async decision; default to sync for the base / custom
        # sessions that don't implement async inference.
        self._use_async = False

    @abstractmethod
    def run(self, inputs: np.ndarray) -> List[np.ndarray]:
        ...

    def run_async(self, inputs: np.ndarray, **kwargs) -> int:
        raise NotImplementedError("Async not supported for this session type")

    def wait(self, job_id: int) -> List[np.ndarray]:
        raise NotImplementedError("Async not supported for this session type")

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        return False
