from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
from loguru import logger


def resolve_embedding_cache_path(
    explicit_path: Optional[str],
    session_path: str,
    filename: str,
) -> Optional[Path]:
    """Resolve explicit cache path or default it beside the model artifact."""
    if explicit_path:
        return Path(explicit_path)
    if not session_path:
        return None

    model_path = Path(session_path)
    cache_dir = model_path if not model_path.suffix else model_path.parent
    if cache_dir == Path("/") or not cache_dir.is_dir():
        return None
    return cache_dir / filename


def load_or_build_embedding(
    explicit_path: Optional[str],
    session_path: str,
    filename: str,
    builder: Callable[[], np.ndarray],
) -> np.ndarray:
    """Load an explicit/discovered cache or build and persist a default cache."""
    cache_path = resolve_embedding_cache_path(explicit_path, session_path, filename)
    if cache_path is not None and (explicit_path or cache_path.is_file()):
        logger.info("Loading pre-computed embedding from '{}'", cache_path)
        return np.load(cache_path).astype(np.float32)

    embedding = np.asarray(builder(), dtype=np.float32)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embedding)
        logger.info("Saved embedding cache to '{}'", cache_path)
    return embedding
