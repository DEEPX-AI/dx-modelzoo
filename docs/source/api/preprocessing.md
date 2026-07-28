# Preprocessing

The preprocessing pipeline in DX-ModelZoo transforms input data into the format required by the model.

**Module:** `dx_modelzoo.preprocessing`

!!! note "See Also"
    - [YAML Configuration](../guides/yaml-config.md) - Preprocessing configuration in YAML
    - [Custom Models](../guides/custom-models.md) - Creating custom preprocessing operations
    - [ModelBuilder](model-builder.md) - How to build preprocessing pipelines

## Overview

### PreprocessingPipeline

A pipeline that sequentially executes preprocessing operations defined in YAML.

```python
class PreprocessingPipeline:
    def __init__(self, steps: list, npu_mode: bool = False)
    def __call__(self, img: np.ndarray) -> np.ndarray
```

| Attribute | Type | Description |
|-----------|------|-------------|
| `steps` | `list` | List of preprocessing operation instances |
| `npu_mode` | `bool` | NPU mode (skips `NPU_SKIP_DEFAULT` operations when True) |

### Usage Example

```python
from dx_modelzoo.loader.model_builder import ModelBuilder

builder = ModelBuilder("resnet50_224x224.yaml")
pipeline = builder.build_preprocessing("onnx")

import cv2
img = cv2.imread("test.jpg")      # (H, W, 3) BGR uint8
result = pipeline(img)             # (1, 3, 224, 224) float32
```

## NPU Skip Behavior

### NPU_SKIP_DEFAULT

```python
NPU_SKIP_DEFAULT = {"div", "normalize", "transpose", "expanddim", "mul", "add", "subtract"}
```

When `npu_mode=True`, operations whose types are included in this set are automatically skipped. This is because the NPU accepts uint8 input and handles these operations internally.

!!! warning "NPU Mode Caution"
    In NPU mode, only spatial transforms such as `resize`, `centercrop`, and `convertcolor` are executed on the CPU. Manually adding arithmetic operations may produce incorrect results.

### NPU Skip Summary

| Type | NPU Skip | Reason |
|------|----------|--------|
| `resize` | ❌ | Spatial transform — requires CPU |
| `centercrop` | ❌ | Spatial transform — requires CPU |
| `convertcolor` | ❌ | Color conversion — requires CPU |
| `div` | ✅ | Handled internally by NPU |
| `normalize` | ✅ | Handled internally by NPU |
| `transpose` | ✅ | Layout conversion handled internally by NPU |
| `expanddim` | ✅ | Batch handling handled internally by NPU |
| `mul` | ✅ | Handled internally by NPU |
| `add` | ✅ | Handled internally by NPU |
| `subtract` | ✅ | Handled internally by NPU |

## Built-in Types

| Type | Summary |
|------|---------|
| `resize` | Resize image |
| `centercrop` | Center crop |
| `convertcolor` | Color conversion |
| `div` | Divide by scalar |
| `normalize` | Mean/std normalization |
| `transpose` | Axis permutation |
| `expanddim` | Insert dimension |
| `mul` | Multiply by scalar |
| `add` | Add scalar |
| `subtract` | Subtract scalar |
| `bgr_to_y_channel` | Convert BGR to Y channel |
| `bgr_to_y_channel_uint8` | Convert BGR to uint8 Y channel |
| `totensor` | HWC uint8 `[0,255]` → CHW float32 `[0,1]` |

### `resize`

Resize an image.

```yaml
- type: resize
  size: [256, 256]          # [H, W]
  mode: torchvision         # torchvision | opencv | letterbox
  interpolation: BILINEAR   # BILINEAR | BICUBIC | NEAREST | AREA
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `size` | `list[int]` | (required) | Target size [H, W] |
| `mode` | `str` | `"torchvision"` | Resize mode |
| `interpolation` | `str` | `"BILINEAR"` | Interpolation method |

### `centercrop`

Center crop.

```yaml
- type: centercrop
  height: 224
  width: 224
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `height` | `int` | Crop height |
| `width` | `int` | Crop width |

### `convertcolor`

Color space conversion.

```yaml
- type: convertcolor
  form: BGR2RGB
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `form` | `str` | Conversion format (`BGR2RGB`, `RGB2BGR`, `BGR2GRAY`, etc.) |

### `div`

Division (scaling).

```yaml
- type: div
  x: 255
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `float` | Divisor value |

!!! note "Automatic NPU Skip"
    `div` is included in `NPU_SKIP_DEFAULT` and is skipped in NPU mode.

### `normalize`

Mean/standard deviation normalization: `(img - mean) / std`

```yaml
- type: normalize
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `mean` | `list[float]` | Per-channel mean |
| `std` | `list[float]` | Per-channel standard deviation |

### `transpose`

Axis transposition (e.g., HWC → CHW).

```yaml
- type: transpose
  axis: [2, 0, 1]
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `axis` | `list[int]` | Axis order |

### `expanddim`

Add a dimension (e.g., batch dimension).

```yaml
- type: expanddim
  axis: 0
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `axis` | `int` | Axis position to insert |

### `mul`

Multiplication.

```yaml
- type: mul
  x: 2.0
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `float` | Multiplier value |

### `add`

Addition.

```yaml
- type: add
  x: 1.0
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `float` | Value to add |

### `subtract`

Subtraction.

```yaml
- type: subtract
  x: 128.0
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `float` | Value to subtract |

### `bgr_to_y_channel`

Convert a BGR image to the Y channel (luminance). Primarily used for super-resolution models.

```yaml
- type: bgr_to_y_channel
```

No parameters.

### `bgr_to_y_channel_uint8`

BGR to Y channel conversion (uint8 output). For NPU models.

```yaml
- type: bgr_to_y_channel_uint8
```

No parameters.

### `totensor`

Convert HWC `uint8` input in `[0, 255]` to CHW `float32` in `[0, 1]`.

```yaml
- type: totensor
```

No parameters.

Equivalent to `div(x=255)` + `transpose`, but registered as a single preprocessing step.

## Custom Preprocessing

```python
from dx_modelzoo.preprocessing import PREPROCESSING_REGISTRY

@PREPROCESSING_REGISTRY.register("my_custom_op")
class MyCustomOp:
    def __init__(self, param: int = 10):
        self.param = param

    def __call__(self, img: np.ndarray) -> np.ndarray:
        # Custom logic
        return img
```

```yaml
preprocessing:
  - type: my_custom_op
    param: 20
```

!!! note "NPU Skip for Custom Operations"
    Custom operations are not skipped in NPU mode by default, as they are not included in `NPU_SKIP_DEFAULT`. If your custom operation is an arithmetic operation that should be skipped on the NPU, you must add it to the set manually.

For detailed usage, refer to the [Custom Models](../guides/custom-models.md) guide.
