from typing import List

import numpy as np


def topk_postprocessing(result: List[np.ndarray], topk=[1, 5]) -> np.ndarray:
    """Perform postprocessing by sorting the model outputs and returning the top-k results.

    This function sorts the model's output results and selects the top-k elements
    based on the provided `topk` values. It can return multiple top-k selections
    if multiple values are specified in the `topk` list.

    Args:
        result (List[np.ndarray]): A list of model output arrays. Each element in
                                   the list should be a numpy array of model predictions.
        topk (list, optional): A list of integers specifying the top-k values
                               to be selected. Defaults to [1, 5]
    Returns:
        np.ndarray: A numpy array containing the sorted and selected top-k elements
                    from the model outputs. The shape and structure of the array
                    will depend on the `topk` values provided.
    """
    result = result[0]
    max_k = max(topk)
    r = np.argsort(result, axis=1)[:, ::-1][:, :max_k]
    return r
