from abc import ABC, abstractmethod
from typing import Tuple

from dx_modelzoo.preprocessing import PreProcessingCompose


class DatasetBase(ABC):
    """Dataset Base Class.
    it needs to dataset root dir path and preprocessing.
    preprocessing is given by Model.
    """

    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir

        self._preprocessing = None

    @property
    def preprocessing(self) -> PreProcessingCompose:
        if self._preprocessing is None:
            raise ValueError("Dataset's Pre Processing is not set.")
        return self._preprocessing

    @preprocessing.setter
    def preprocessing(self, preprocessing: PreProcessingCompose) -> None:
        self._preprocessing = preprocessing

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple:
        ...


__all__ = ["DatasetBase"]
