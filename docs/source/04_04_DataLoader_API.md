# DataLoader

DX-ModelZoo provides a pure Python/NumPy DataLoader for efficient batch loading and preprocessing of datasets.

**Module:** `dx_modelzoo.common.dataloader`

## DatasetBase

An abstract base class that all datasets must inherit from.

```python
class DatasetBase(ABC):
    def __init__(self, data_dir: str)

    @property
    def preprocessing(self) -> Optional[Callable]
        """Preprocessing pipeline getter"""

    @preprocessing.setter
    def preprocessing(self, value: Callable) -> None
        """Preprocessing pipeline setter"""

    @abstractmethod
    def __len__(self) -> int
        """Return the dataset size"""

    @abstractmethod
    def __getitem__(self, idx: int) -> tuple
        """Return a sample by index"""

    @abstractmethod
    def validate(self) -> None
        """Validate dataset files and directory structure"""
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `data_dir` | `str` | Path to the data directory |
| `_preprocessing` | `Callable \| None` | Preprocessing pipeline (set by the Evaluator) |

`DatasetBase.preprocessing` raises `ValueError` until an Evaluator assigns a
preprocessing pipeline. Custom datasets can inspect `_preprocessing` before
applying it in `__getitem__`.

### Usage Example

```python
from dx_modelzoo.common.dataloader import DatasetBase

class MyDataset(DatasetBase):
    def __init__(self, data_dir: str):
        super().__init__(data_dir)
        self.files = sorted(Path(data_dir).glob("*.jpg"))

    def validate(self) -> None:
        # Required abstract method — without it the class cannot be instantiated
        self.ensure_exists(self.data_dir, "Place *.jpg files under this directory.")
        if not self.files:
            raise RuntimeError(f"No *.jpg files found under {self.data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.files[idx]))
        if self.preprocessing:
            img = self.preprocessing(img)
        return img, idx
```

!!! warning  
    `__len__`, `__getitem__` **and** `validate` are abstract. Omitting `validate`
    raises `TypeError: Can't instantiate abstract class MyDataset without an
    implementation for abstract method 'validate'`.

## DataLoader

An iterator that traverses a dataset in batches.

```python
class DataLoader:
    def __init__(
        self,
        dataset: DatasetBase,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        collate_fn: Optional[Callable] = None,
        prefetch_factor: int = 2,
        start_method: str = "auto",
    )
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | `DatasetBase` | (required) | Dataset instance |
| `batch_size` | `int` | `1` | Batch size |
| `shuffle` | `bool` | `False` | Whether to shuffle the data |
| `num_workers` | `int` | `0` | Number of parallel loading workers (0 = main process) |
| `collate_fn` | `Callable \| None` | `None` | Batch collation function (None = default numpy collate) |
| `prefetch_factor` | `int` | `2` | Number of batches to prefetch per worker |
| `start_method` | `str` | `"auto"` | Multiprocessing method: `auto`, `forkserver`, or `spawn` |

### Methods

```python
def __len__(self) -> int
    """Return the total number of batches"""

def __iter__(self) -> Iterator
    """Batch iterator"""
```

### Usage Example

```python
from dx_modelzoo.common.dataloader import DataLoader

loader = DataLoader(
    dataset=my_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=4,
    prefetch_factor=2,
)

print(len(loader))  # Total number of batches

for batch in loader:
    # Default collate does not stack tuple samples:
    #   batch_size == 1 -> the sample itself, e.g. (img, label)
    #   batch_size > 1  -> list of samples, e.g. [(img, label), ...]
    for img, label in batch:
        # img: np.ndarray (3, 224, 224), label: int
        results = session.run(np.expand_dims(img, 0))
```

!!! note "Default collate behavior"  
    Only `Mapping` samples are merged into batched numpy arrays. Tuple samples
    are returned as a Python `list` (or the single sample when `batch_size=1`),
    so integer labels stay `int` inside a `list` — they are **not** converted to
    `np.ndarray`. Pass a custom `collate_fn` if you need stacked arrays.

### Single / Multi-Process Modes

=== "Single Process (num_workers=0)"

    ```python
    loader = DataLoader(dataset, batch_size=4, num_workers=0)
    ```

    - Loads data sequentially in the main process
    - Useful for debugging
    - May become a bottleneck for I/O-bound workloads

=== "Multi-Process (num_workers>0)"

    ```python
    loader = DataLoader(dataset, batch_size=4, num_workers=4, prefetch_factor=2)
    ```

    - Loads data in parallel across separate processes
    - Use `prefetch_factor` to control how many batches are prepared in advance
    - Maximizes throughput for large-scale datasets

## `make_dataloader`

A factory function for conveniently creating a DataLoader.

```python
def make_dataloader(
    dataset: DatasetBase,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    collate_fn: Optional[Callable] = None,
    start_method: str = "auto",
) -> DataLoader
```

```python
from dx_modelzoo.common.dataloader import make_dataloader

loader = make_dataloader(dataset, batch_size=8, num_workers=4)
```

## Batch Collation

The default collate function stacks NumPy arrays along the batch axis:

```python
def _numpy_collate(batch: list[tuple]) -> tuple:
    """Merge a list of samples into a batch tensor

    [(img1, label1), (img2, label2), ...]
    → (np.stack([img1, img2, ...]), np.array([label1, label2, ...]))
    """
```

You can also provide a custom collate function:

```python
def my_collate(batch):
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    # Handle variable-size images, etc.
    return images, labels

loader = DataLoader(dataset, batch_size=4, collate_fn=my_collate)
```

## Differences from PyTorch

The following table summarizes the key differences:

| Feature | DX ModelZoo DataLoader | PyTorch DataLoader |
|---------|----------------------|-------------------|
| Dependencies | Pure Python/NumPy (DataLoader), torchvision for preprocessing fusion | Requires PyTorch |
| Tensor type | `np.ndarray` | `torch.Tensor` |
| Default collate | `np.stack` | `torch.stack` |
| pin_memory | ❌ | ✅ |
| sampler | ❌ | ✅ |
| drop_last | ❌ | ✅ |
| Multi-process | Python `multiprocessing` | Custom implementation |

!!! note "Design Philosophy"  
    DX-ModelZoo is designed for lightweight deployment in NPU environments. The DataLoader is implemented in pure Python/NumPy. PyTorch is NOT required for the DataLoader itself, though torchvision is used by the preprocessing pipeline for dx_com fusion compatibility.

## Integration with Evaluator

In most cases, you do not need to create a DataLoader directly. `EvaluatorBase` internally calls `make_loader()` to create one automatically:

```python
class EvaluatorBase:
    def make_loader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
        )
```

If direct usage is required (e.g., for a custom evaluation loop):

```python
from dx_modelzoo.loader.model_builder import ModelBuilder
from dx_modelzoo.common.dataloader import make_dataloader

builder = ModelBuilder("resnet50_224x224.yaml")
dataset = builder.build_dataset()
preprocessing = builder.build_preprocessing("onnx")
dataset.preprocessing = preprocessing

loader = make_dataloader(dataset, batch_size=8, num_workers=4)

for images, labels in loader:
    output = session.run(images)
    # Custom processing ...
```
