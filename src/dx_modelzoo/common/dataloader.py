from __future__ import annotations

import multiprocessing as mp
import os
import random
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterator, List, Optional, Tuple

import numpy as np
from loguru import logger


class DatasetBase(ABC):
    """Abstract base class for all datasets."""

    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir
        self._preprocessing = None
        self.transform = None  # torchvision Compose or Dict[str, Compose] for dx_com fusion

    @property
    def preprocessing(self):
        if self._preprocessing is None:
            raise ValueError("Dataset's preprocessing is not set.")
        return self._preprocessing

    @preprocessing.setter
    def preprocessing(self, value):
        self._preprocessing = value
        # PreprocessingPipeline IS a Compose subclass — assign directly.
        # dx_com's _is_compose() will recognize it natively.
        from torchvision.transforms import Compose

        if isinstance(value, Compose):
            self.transform = value
        elif hasattr(value, "compose"):
            # Legacy fallback
            self.transform = value.compose
        else:
            self.transform = None

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple:
        ...

    def ensure_exists(self, data_dir: str, install_guide: str) -> None:
        """Raise ``FileNotFoundError`` with detailed install instructions if ``data_dir`` is missing.

        Args:
            data_dir: Full path to the dataset directory.
            install_guide: Multi-line installation instructions.
        """
        if os.path.isdir(data_dir):
            return

        msg = (
            f"\n{'=' * 70}\n"
            f"  Dataset not found: {self.__class__.__name__}\n"
            f"  Expected path: {data_dir}\n"
            f"{'=' * 70}\n\n"
            f"{install_guide}\n\n"
            f"After downloading, set the DATA_ROOT environment variable or\n"
            f"use --data-root to point to the datasets directory.\n"
            f"{'=' * 70}\n"
        )
        raise FileNotFoundError(msg)


def _worker_loop(dataset, index_queue, result_queue, worker_seed=None):
    """Worker process: fetch indices from index_queue, put results to result_queue."""
    if worker_seed is not None:
        random.seed(worker_seed)
        np.random.seed(worker_seed % (2**32))
    while True:
        order_idx = index_queue.get()
        if order_idx is None:
            break
        order, idx = order_idx
        try:
            data = dataset[idx]
            result_queue.put((order, data))
        except Exception as e:
            import traceback

            e.__traceback_str__ = traceback.format_exc()
            result_queue.put((order, e))


def _numpy_collate(batch):
    """Default collate: stack numpy arrays along a new batch axis.

    Each item in batch is a tuple of (array, label, ...).
    Returns a tuple where arrays are stacked with np.stack.
    """
    if not batch:
        return batch
    elem = batch[0]
    if isinstance(elem, tuple):
        lengths = [len(b) for b in batch]
        n = min(lengths)
        if max(lengths) != n:
            import warnings

            warnings.warn(
                f"Inconsistent tuple lengths in batch collation: {set(lengths)}. " f"Truncating to {n} elements.",
                stacklevel=2,
            )
        return tuple(
            np.stack([b[i] for b in batch]) if isinstance(batch[0][i], np.ndarray) else [b[i] for b in batch]
            for i in range(n)
        )
    if isinstance(elem, np.ndarray):
        return np.stack(batch)
    return batch


class DataLoader:
    """Pure-Python DataLoader with optional multi-process prefetching.

    Supports batch_size > 1 with numpy-based collation (no torch).
    """

    def __init__(
        self,
        dataset: DatasetBase,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        collate_fn: Optional[Callable] = None,
        prefetch_factor: int = 2,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = min(num_workers, os.cpu_count() or 1)
        self.collate_fn = collate_fn or (_numpy_collate if batch_size > 1 else None)
        self.prefetch_factor = prefetch_factor

    def __len__(self) -> int:
        n = len(self.dataset)
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator:
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)

        if self.num_workers <= 0:
            yield from self._iter_single(indices)
        else:
            yield from self._iter_multi(indices)

    def _collate_batch(self, items: List) -> Any:
        if self.collate_fn is not None:
            return self.collate_fn(items)
        return items[0] if len(items) == 1 else items

    def _iter_single(self, indices: List[int]) -> Iterator:
        """Single-process sequential iteration with batching."""
        batch = []
        for idx in indices:
            batch.append(self.dataset[idx])
            if len(batch) == self.batch_size:
                yield self._collate_batch(batch)
                batch = []
        if batch:
            yield self._collate_batch(batch)

    def _iter_multi(self, indices: List[int]) -> Iterator:
        """Multi-process prefetched iteration using mp.Process + mp.Queue."""
        from dx_modelzoo.common.seed import get_seed

        base_seed = get_seed()
        ctx = mp.get_context("forkserver")
        index_queue = ctx.Queue()
        result_queue = ctx.Queue()

        workers = []
        for wid in range(self.num_workers):
            worker_seed = (base_seed + wid) if base_seed is not None else None
            w = ctx.Process(
                target=_worker_loop,
                args=(self.dataset, index_queue, result_queue, worker_seed),
                daemon=True,
            )
            w.start()
            workers.append(w)

        # Fill the index queue with prefetch_factor * num_workers items ahead
        prefetch_total = self.prefetch_factor * self.num_workers
        send_idx = 0
        prefetch_count = min(prefetch_total, len(indices))
        for i in range(prefetch_count):
            index_queue.put((i, indices[i]))
        send_idx = prefetch_count

        # Collect results in order, then batch
        pending = {}
        next_order = 0
        batch = []

        try:
            for _ in range(len(indices)):
                while next_order not in pending:
                    order, data = result_queue.get()
                    if isinstance(data, Exception):
                        if hasattr(data, "__traceback_str__"):
                            logger.error("Worker error:\n{}", data.__traceback_str__)
                        raise data
                    pending[order] = data

                batch.append(pending.pop(next_order))
                next_order += 1

                if len(batch) == self.batch_size:
                    yield self._collate_batch(batch)
                    batch = []

                # Enqueue next index
                if send_idx < len(indices):
                    index_queue.put((send_idx, indices[send_idx]))
                    send_idx += 1

            if batch:
                yield self._collate_batch(batch)
        finally:
            # Drain queues so workers aren't blocked on full result_queue
            # or waiting on index_queue.get()
            try:
                while not result_queue.empty():
                    result_queue.get_nowait()
            except Exception:
                pass
            try:
                while not index_queue.empty():
                    index_queue.get_nowait()
            except Exception:
                pass
            # Send stop sentinels
            for _ in workers:
                try:
                    index_queue.put_nowait(None)
                except Exception:
                    pass
            # Short join, then force-terminate
            for w in workers:
                w.join(timeout=1)
                if w.is_alive():
                    w.terminate()
                    w.join(timeout=1)


def make_dataloader(
    dataset: DatasetBase,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    collate_fn: Optional[Callable] = None,
) -> DataLoader:
    """Create a DataLoader wrapping a DatasetBase.

    Uses identity collate by default (batch_size=1, no stacking)
    so that numpy arrays from __getitem__ are passed through as-is.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
