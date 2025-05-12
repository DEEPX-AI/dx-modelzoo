from datetime import datetime

import numpy as np
import torch

from dx_modelzoo.exceptions import UnKnownError


class EvaluationTimer:
    def __init__(self) -> None:
        self.start_time = None

    def __enter__(self) -> None:
        print("Evaluation Start.\n")
        self.start_time = datetime.now()

    def __exit__(self, exc_type: Exception, *args, **kwargs) -> None:
        if exc_type is not None:
            raise UnKnownError() from exc_type
        end_time = datetime.now()
        print(f"Total RunTime {str(end_time - self.start_time)}")


def torch_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()
