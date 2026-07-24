from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.dataset import DATASET_REGISTRY

_DTYPE_MAP = {
    "float32": np.float32,
    "float16": np.float16,
    "float64": np.float64,
    "int64": np.int64,
    "int32": np.int32,
    "uint8": np.uint8,
}


@DATASET_REGISTRY.register
class SyntheticMultiInput(DatasetBase):
    """Synthetic multi-input dataset for throughput / load validation.

    Produces ``num_samples`` items, each a dict of named tensors built
    from the model's ``inputs`` YAML spec.  Used by models whose
    inputs are pre-formatted tensors (BlazeFace control scalars,
    FoundationPose RGB-D pairs, PointPillars pseudo_image,
    DiffusionPolicy noisy_action / timestep / obs_cond) where no
    real-image dataset is available in the current environment.

    Args:
        data_dir: Ignored; present for ``DatasetBase`` contract.
        inputs:   List of ``{name, shape, dtype}`` dicts mirroring the
                  YAML ``inputs`` block.  Injected by the YAML loader.
        num_samples: Number of synthetic samples to generate.
        seed:     Base RNG seed; per-sample seeding keeps results
                  reproducible across processes.
    """

    def __init__(
        self,
        data_dir: str = "",
        inputs: List[Dict] = None,
        num_samples: int = 256,
        seed: int = 0,
    ) -> None:
        super().__init__(data_dir or "")
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.inputs_spec: List[Dict] = inputs or []

    def __len__(self) -> int:
        return self.num_samples

    def _make_tensor(self, rng: np.random.Generator, spec: Dict):
        shape = tuple(spec.get("shape", [1]))
        dtype = _DTYPE_MAP.get(str(spec.get("dtype", "float32")), np.float32)
        if "value" in spec:
            return np.full(shape, spec["value"], dtype=dtype)
        if np.issubdtype(dtype, np.integer):
            return rng.integers(0, 1000, size=shape).astype(dtype)
        return rng.standard_normal(shape).astype(dtype)

    def __getitem__(self, idx: int) -> Tuple:
        rng = np.random.default_rng(self.seed + idx + 1)
        sample: Dict[str, np.ndarray] = {}
        for spec in self.inputs_spec:
            sample[str(spec["name"])] = self._make_tensor(rng, spec)
        return sample, idx
