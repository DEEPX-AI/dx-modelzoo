"""Global seed utilities for reproducible evaluation."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
from loguru import logger

_global_seed: Optional[int] = None


def set_seed(seed: int) -> None:
    """Set global seed for all random number generators.

    Seeds: Python ``random``, ``numpy``, and ``torch`` (if available).
    Also sets environment variables for hash seed and CUBLAS determinism.
    """
    global _global_seed
    _global_seed = seed

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    except ImportError:
        pass

    logger.info(f"Global seed set to {seed}")


def get_seed() -> Optional[int]:
    """Return the current global seed, or ``None`` if not set."""
    return _global_seed
