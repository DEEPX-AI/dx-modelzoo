from abc import ABC, abstractmethod

import numpy as np

from dx_modelzoo.enums import SessionType


class SessionBase(ABC):
    """Session Base Class.
    it needs file path, and session type.
    """

    def __init__(self, path: str, session_type: SessionType) -> None:
        self.path = path
        self.type = session_type

    @abstractmethod
    def run(self, inputs: np.ndarray) -> np.ndarray:
        """run session.

        Args:
            inputs (np.ndarray): input value.

        Returns:
            np.ndarray: session output value.
        """
        ...


__all__ = ["SessionBase"]
