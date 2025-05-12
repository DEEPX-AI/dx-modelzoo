from abc import ABC, abstractmethod
from typing import Callable, Dict, List

from torch.utils.data import DataLoader

from dx_modelzoo.dataset import DatasetBase
from dx_modelzoo.preprocessing import PreProcessingCompose
from dx_modelzoo.session import SessionBase

PREPROCESSING = Dict[str, Dict[str, str | int | float | List[int | float]]]


class EvaluatorBase(ABC):
    """Evaluator Base Class.
    it needs session, dataset, and postprocessing.
    postprocessing is given by model.
    """

    def __init__(self, session: SessionBase, dataset: DatasetBase):
        self.session = session
        self.dataset = dataset
        self._postprocessing = None

    @property
    def postprocessing(self) -> Callable:
        if self._postprocessing is None:
            raise ValueError("Evaluator's Post Processing is not set.")
        return self._postprocessing

    @abstractmethod
    def eval(self):
        ...

    def set_preprocessing(self, preprocessings: List[PREPROCESSING]) -> None:
        """set preprocessing.

        Args:
            preprocessings (List[PREPROCESSING]): preprocessing list.
        """
        self.dataset.preprocessing = PreProcessingCompose(preprocessings, self.session.type)

    def set_postprocessing(self, postprocessing: Callable) -> None:
        """set postprocessing.

        Args:
            postprocessing (Callable): postprocessing func.
        """
        self._postprocessing = postprocessing

    def make_loader(self) -> DataLoader:
        """make data loader from dataset.

        Returns:
            DataLoader: dataloader.
        """
        return DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=4)


__all__ = ["EvaluatorBase"]
