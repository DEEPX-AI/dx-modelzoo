from typing import Callable, Dict, List

import numpy as np
from torchvision.transforms import Compose

from dx_modelzoo.preprocessing.centercrop import CenterCrop
from dx_modelzoo.preprocessing.convertcolor import ConvertColor
from dx_modelzoo.preprocessing.div import Div
from dx_modelzoo.preprocessing.expanddim import ExpandDim
from dx_modelzoo.preprocessing.normalize import Normalize
from dx_modelzoo.preprocessing.resize import Resize
from dx_modelzoo.preprocessing.transpose import Transpose

PREPROCESSING = Dict[str, Dict[str, str | int | float | List[int | float]]]


PREPROCESSING_MAP = {
    "resize": Resize,
    "centercrop": CenterCrop,
    "convertColor": ConvertColor,
    "normalize": Normalize,
    "div": Div,
    "transpose": Transpose,
    "expandDim": ExpandDim,
}


def parse_ort_preprocessing_ops(preprocessings: List[PREPROCESSING]) -> List[Callable]:
    """parse onnx runtime preprocessings.
    if expandDim preprocessing in preprocessing appear just one time, it removed.

    Args:
        preprocessings (List[PREPROCESSING]): preprocessing list.

    Returns:
        List[Callable]: preprocessing ops list.
    """
    expand_dim = True
    ops_list = []
    for preprocessing in preprocessings:
        ops_name = list(preprocessing.keys())[0]
        if ops_name == "expandDim" and expand_dim is True:
            expand_dim = False
            continue
        ops = PREPROCESSING_MAP[ops_name]
        kwargs = preprocessing[ops_name]
        ops_list.append(ops(**kwargs))
    return ops_list


def parse_npu_preprocessing_ops(preprocessings: List[PREPROCESSING]) -> List[Callable]:
    """parse npu preprocessing ops.
    it removes calculation operations like div, mul, add, normalizations.
    and also it remove transpose operations.

    Args:
        preprocessings (List[PREPROCESSING]): preprocessing list.

    Returns:
        List[Callable]: preprocessing ops list.
    """
    expand_dim = True
    ops_list = []
    for preprocessing in preprocessings:
        ops_name = list(preprocessing.keys())[0]
        if ops_name == "expandDim" and expand_dim is True:
            expand_dim = False
            continue

        if ops_name in ["div", "mul", "add", "normalize", "subtract"]:
            continue
        ops = PREPROCESSING_MAP[ops_name]
        kwargs = preprocessing[ops_name]
        ops_list.append(ops(**kwargs))
    return ops_list


def parse_preprocessing_ops(preprocessings: List[PREPROCESSING], session_type: str) -> List[Callable]:
    """parse preprocessing ops.

    Args:
        preprocessings (List[PREPROCESSING]): preprocessings.

    Returns:
        List[Callable]: preprocessings ops list.
    """
    if session_type == "OnnxRuntime":
        return parse_ort_preprocessing_ops(preprocessings)
    elif session_type == "DxRuntime":
        return parse_npu_preprocessing_ops(preprocessings)
    else:
        raise ValueError(f"Invalid sessiont type: {session_type}")


class PreProcessingCompose:
    """Pre Processing Compose.
    it composes pre processing operations.

    Args:
        preprocessings (List[PREPROCESSING]): preprocessings List.
    """

    def __init__(self, preprocessings: List[PREPROCESSING], session_type: str) -> None:
        self._check_preprocessings(preprocessings)
        self.preprocessings_ops = parse_preprocessing_ops(preprocessings, session_type)

    def _check_preprocessings(self, preprocessings: List[PREPROCESSING]) -> None:
        """check if the preprocessings are valid.

        Args:
            preprocessings (List[PREPROCESSING]): preprocessings.

        Raises:
            ValueError: if the preprocessings are invalid, raise ValueError.
        """
        for preprocessing in preprocessings:
            ops_name = list(preprocessing.keys())[0]
            if ops_name not in PREPROCESSING_MAP:
                raise ValueError(f"Invalid Preprocessing name. {preprocessing}")

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """Pre processing the inputs.

        Args:
            inputs (np.ndarray): inputs.

        Returns:
            np.ndarray: pre processed inputs.
        """
        compose = Compose(self.preprocessings_ops)
        return compose(inputs)
