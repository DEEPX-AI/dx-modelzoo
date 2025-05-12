from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from dx_modelzoo.enums import DatasetType, EvaluationType
from dx_modelzoo.evaluator import EvaluatorBase
from dx_modelzoo.utils import EvaluationTimer

PREPROCESSING = Dict[str, Dict[str, str | int | float | List[int | float]]]


def make_indent_string(string: str, num_indent: int = 2) -> str:
    indent = " " * num_indent
    return f"{indent}{string}"


@dataclass
class ModelInfo:
    name: str
    dataset: DatasetType
    evaluation: EvaluationType
    raw_performance: Optional[str] = None
    q_lite_performance: Optional[str] = None
    q_pro_performance: Optional[str] = None
    q_master_performance: Optional[str] = None

    def print(self) -> None:
        print("Model Information")

        name = make_indent_string(f"Name: {self.name}")
        dataset = make_indent_string(f"Dataset: {self.dataset}")
        evaluation = make_indent_string(f"Evaluation method: {self.evaluation}")
        metric = make_indent_string(f"Evaluation metric: {self.evaluation.metric()}")
        raw_performance = make_indent_string(f"Raw performance: {self.raw_performance}")
        q_lite_performance = make_indent_string(f"Q-Lite performance: {self.q_lite_performance}")
        q_pro_performance = make_indent_string(f"Q-Pro performance: {self.q_pro_performance}")
        q_master_performance = make_indent_string(f"Q-Master performance: {self.q_master_performance}")

        print(
            f"{name}\n{dataset}\n{evaluation}\n{metric}\n{raw_performance}\n{q_lite_performance}\n{q_pro_performance}\n"
            f"{q_master_performance}"
        )


class ModelBase(ABC):
    """Model Base class.
    it needs evaluator.
    """

    @property
    @abstractmethod
    def info(cls) -> ModelInfo:
        """Model info property.

        Returns:
            ModelInfo: model info.
        """
        pass

    def __init__(self, evaluator: EvaluatorBase) -> None:
        self.evaluator = evaluator

        self.evaluator.set_preprocessing(self.preprocessing())
        self.evaluator.set_postprocessing(self.postprocessing())

    def eval(self) -> None:
        """evalutation model."""
        with EvaluationTimer():
            self.evaluator.eval()

    @classmethod
    def print_info(cls):
        """print model info."""
        cls.info.print()

    @abstractmethod
    def preprocessing(self) -> List[PREPROCESSING]:
        """model prerpocessing.
        if has same form of dx-com config's preprocessing.

        Returns:
            List[PREPROCESSING]: prerpocessings.
        """
        pass

    @abstractmethod
    def postprocessing(self) -> Callable:
        """model postprocessing.
        it return postprocessing func.

        Returns:
            Callable: postprocessing func.
        """
        pass


__all__ = ["ModelBase"]
