# Custom Datasets

This guide explains how to add a custom dataset to DX ModelZoo. Inherit from `DatasetBase`, register it with the Registry, and reference it directly in your YAML configuration.

## Overview

```mermaid
graph LR
    A[DatasetBase ABC] --> B[Custom Dataset]
    B --> C[DATASET_REGISTRY.register]
    C --> D[Reference in YAML]
    D --> E[DataLoader auto-uses it]
```

## DatasetBase Interface

All datasets must inherit from the `DatasetBase` abstract class:

```python
from dx_modelzoo.common.dataloader import DatasetBase

class DatasetBase(ABC):
    def __init__(self, data_dir: str):
        self._data_dir = data_dir
        self._preprocessing = None

    @property
    def preprocessing(self):
        """Preprocessing pipeline (automatically set by the Evaluator)"""
        return self._preprocessing

    @preprocessing.setter
    def preprocessing(self, value):
        self._preprocessing = value

    @abstractmethod
    def __len__(self) -> int:
        """Return the size of the dataset"""
        ...

    @abstractmethod
    def __getitem__(self, idx: int) -> tuple:
        """Return a sample by index"""
        ...
```

## Implementation

### Step 1: Write the Dataset Class

```python title="src/dx_modelzoo/dataset/my_dataset.py"
import os
from pathlib import Path

import cv2
import numpy as np

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.dataset import DATASET_REGISTRY


@DATASET_REGISTRY.register("my_custom_dataset")
class MyCustomDataset(DatasetBase):
    """Custom image classification dataset

    Directory structure:
        data_dir/
        ├── images/
        │   ├── 00001.jpg
        │   └── ...
        └── labels.txt        # "filename label" format
    """

    def __init__(self, data_dir: str):
        super().__init__(data_dir)
        self._samples = self._load_annotations()

    def _load_annotations(self) -> list[tuple[str, int]]:
        """Parse the annotation file and return a list of (image_path, label) tuples"""
        label_file = Path(self._data_dir) / "labels.txt"
        samples = []
        with open(label_file, "r") as f:
            for line in f:
                filename, label = line.strip().split()
                img_path = os.path.join(self._data_dir, "images", filename)
                samples.append((img_path, int(label)))
        return samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple:
        img_path, label = self._samples[idx]

        # Load image (BGR format)
        img = cv2.imread(img_path)

        # Apply preprocessing (pipeline set by the Evaluator)
        if self.preprocessing is not None:
            img = self.preprocessing(img)

        return img, label
```

!!! info "Where to Apply `preprocessing`"
    `self.preprocessing` is automatically configured by the Evaluator via `set_preprocessing()`. The standard pattern is to apply it in `__getitem__` after loading the image.

## Registration

### Reference in YAML

```yaml title="MyModel.yaml"
dataset:
  type: my_custom_dataset       # must match the register() name
  eval_path: ${DATA_ROOT}/my_data
```

### Run Evaluation

```bash
dxmz eval MyModel --profile onnx --data-root /path/to/datasets
```

## YAML Configuration

The return format of `__getitem__` must match the Evaluator's `extract_inputs` method:

| Evaluator | Return Format | Description |
|-----------|---------------|-------------|
| `image_classification` | `(image, label)` | label: int |
| `coco` | `(image, annotation)` | annotation: dict (COCO format) |
| `segmentation` | `(image, mask)` | mask: np.ndarray |
| `depth_estimation` | `(image, depth_map)` | depth_map: np.ndarray |
| `face_detection` | `(image, boxes)` | boxes: np.ndarray |

!!! tip "Compatibility with Built-in Evaluators"
    To use a built-in Evaluator, your dataset's return format must match exactly what that Evaluator's `extract_inputs` expects.

## Multi-Input Datasets

A full example implementing a COCO-format custom dataset:

```python title="src/dx_modelzoo/dataset/my_coco.py"
import json
from pathlib import Path

import cv2
import numpy as np

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.dataset import DATASET_REGISTRY


@DATASET_REGISTRY.register("my_coco_format")
class MyCocoDataset(DatasetBase):
    """Custom dataset in COCO JSON format"""

    def __init__(self, data_dir: str):
        super().__init__(data_dir)
        anno_path = Path(data_dir) / "annotations.json"
        with open(anno_path) as f:
            self._coco = json.load(f)
        self._images = self._coco["images"]
        self._build_index()

    def _build_index(self):
        """Build image_id to annotations mapping"""
        self._img_to_anns = {}
        for ann in self._coco["annotations"]:
            img_id = ann["image_id"]
            self._img_to_anns.setdefault(img_id, []).append(ann)

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int) -> tuple:
        img_info = self._images[idx]
        img_path = Path(self._data_dir) / img_info["file_name"]
        img = cv2.imread(str(img_path))

        if self.preprocessing is not None:
            img = self.preprocessing(img)

        annotation = {
            "image_id": img_info["id"],
            "annotations": self._img_to_anns.get(img_info["id"], []),
            "width": img_info["width"],
            "height": img_info["height"],
        }
        return img, annotation
```

```yaml title="Corresponding YAML configuration"
dataset:
  type: my_coco_format
  eval_path: ${DATA_ROOT}/my_coco_data

evaluator:
  type: coco
```

### DataLoader Integration

Datasets are automatically batched by the `DataLoader`:

```python
from dx_modelzoo.common.dataloader import make_dataloader

dataset = MyCustomDataset("/path/to/data")
loader = make_dataloader(dataset, batch_size=4, shuffle=False, num_workers=4)

for batch in loader:
    images, labels = batch
    # images: np.ndarray (batch_size, ...)
```

!!! warning "NumPy-Based"
    The DataLoader is implemented in pure Python/NumPy. PyTorch is NOT required for the DataLoader itself, though torchvision is used by the preprocessing pipeline for dx_com fusion compatibility. Ensure that `__getitem__` returns NumPy arrays.

## Troubleshooting

| Item | Check |
|------|-------|
| Does the class inherit from `DatasetBase`? | ✅ |
| Are `__len__` and `__getitem__` implemented? | ✅ |
| Is the class registered with `@DATASET_REGISTRY.register()`? | ✅ |
| Does `__getitem__` apply `self.preprocessing`? | ✅ |
| Does `dataset.type` in YAML match the registered name? | ✅ |
| Is the return format compatible with the Evaluator's `extract_inputs`? | ✅ |
