# YAML Model Configuration

Every model in DX ModelZoo is defined by a single YAML file. No Python code is needed — preprocessing, postprocessing, evaluation, and compilation settings are all included.

## Full Structure

```yaml
name: resnet50_224x224                # Model name
task: image_classification    # Task type
reference:                    # Paper/repository URL
description:                  # Model description (multiline)

inputs:                       # Input tensor definitions
  - name: input
    shape: [1, 3, 224, 224]
    dtype: float32
    layout: NCHW

preprocessing:                # Preprocessing pipeline
  - type: resize
    size: [256, 256]
  - type: centercrop
    height: 224
    width: 224
  # ...

postprocessing:               # Postprocessing pipeline
  - type: topk
    k: [1, 5]

evaluator:                    # Evaluator type
  type: image_classification

dataset:                      # Dataset configuration
  type: ILSVRC2012
  eval_path: ${DATA_ROOT}/ILSVRC2012/val

artifacts:                    # Model file paths
  path: ${MODEL_ROOT}/${MODEL_NAME}

profiles:                     # Runtime profiles
  onnx:
    target: onnx
    runtime:
      device: gpu
      batch_size: 1
  q-lite:
    target: dxnn
    compile:
      quantization: lite
    runtime:
      device: 0
```

## Section Details

### `name` / `task`

```yaml
name: YOLOv8s              # Unique name (should match filename)
task: object_detection      # Task type
```

Supported tasks:

| Task | Description |
|------|-------------|
| `image_classification` | Image classification |
| `object_detection` | Object detection |
| `semantic_segmentation` | Semantic segmentation |
| `instance_segmentation` | Instance segmentation |
| `face_detection` | Face detection |
| `face_landmark` | Facial landmark detection |
| `pose_estimation` | Pose estimation |
| `depth_estimation` | Depth estimation |
| `super_resolution` | Super resolution |
| `oriented_object_detection` | Oriented bounding box detection (OBB) |

### `inputs`

Defines the model's input tensors. Multi-input models are supported.

```yaml
# Single input (most CV models)
inputs:
  - name: input
    shape: [1, 3, 224, 224]
    dtype: float32
    layout: NCHW

# Multiple inputs (e.g., BERT NLP models)
inputs:
  - name: input_ids
    shape: [1, 256]
    dtype: int64
  - name: attention_mask
    shape: [1, 256]
    dtype: int64
  - name: token_type_ids
    shape: [1, 256]
    dtype: int64
```

### `preprocessing`

The preprocessing pipeline applied to input data. Steps execute in order.

```yaml
preprocessing:
  - type: resize
    mode: torchvision
    size: [256, 256]
    interpolation: BILINEAR
  - type: centercrop
    height: 224
    width: 224
  - type: convertcolor
    form: BGR2RGB
  - type: div
    x: 255
  - type: normalize
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
  - type: transpose
    axis: [2, 0, 1]
  - type: expanddim
    axis: 0
```

Built-in preprocessing types:

| Type | Parameters | Description |
|------|------------|-------------|
| `resize` | `size`, `mode`, `interpolation` | Image resize |
| `centercrop` | `height`, `width` | Center crop |
| `convertcolor` | `form` | Color conversion (e.g., BGR2RGB) |
| `div` | `x` | Divide values (normalization) |
| `normalize` | `mean`, `std` | Mean/std normalization |
| `transpose` | `axis` | Axis permutation (HWC→CHW) |
| `expanddim` | `axis` | Add dimension (batch) |
| `bgr_to_y_channel` | — | BGR → Y channel conversion |

!!! tip "Automatic Skip in NPU Mode"
    When evaluating with a `target: dxnn` profile, arithmetic operations (`div`, `normalize`, `transpose`, `expanddim`) are automatically skipped.
    The NPU handles these operations internally.

### `postprocessing`

The postprocessing pipeline applied to model outputs.

```yaml
# Classification
postprocessing:
  - type: topk
    k: [1, 5]

# Object Detection
postprocessing:
  - type: nms
    variant: yolo

# Segmentation
postprocessing:
  - type: segmentation_argmax

# No postprocessing
postprocessing:
  - type: identity
```

### `dataset`

Dataset configuration for evaluation. Use `${DATA_ROOT}` to specify the root path.

```yaml
dataset:
  type: ILSVRC2012
  eval_path: ${DATA_ROOT}/ILSVRC2012/val
```

Built-in datasets:

| Type | Task | Example Path |
|------|------|-------------|
| `ILSVRC2012` | Classification | `ILSVRC2012/val` |
| `COCO` | Detection | `COCO/official` |
| `PascalVOC2007` | Detection | `VOCdevkit/VOC2007` |
| `Cityscapes` | Segmentation | `cityscapes` |
| `ADE20K` | Segmentation | `ADEChallengeData2016` |
| `WiderFace` | Face detection | `WIDER_FACE` |
| `COCOPose` | Pose estimation | `COCO/official` |
| `NYUDepthv2` | Depth estimation | `NYU_Depth_V2` |
| `BSD68` | Denoising | `BSD68` |
| `BSD100` | Super resolution | `BSD100` |
| `DOTAv1` | Oriented detection | `DOTA/val` |
| `LFW` | Face verification | `lfw` |

### `profiles`

Define multiple runtime profiles. Each profile specifies its own target, compilation options, and runtime options.

```yaml
profiles:
  # ONNX Runtime profile
  onnx:
    target: onnx
    runtime:
      device: gpu           # gpu or cpu
      batch_size: 4         # Batch size (adjustable for onnx only)
      async: false          # Default for onnx profiles

  # NPU Compile-Lite (opt_level=0)
  c-lite:
    target: dxnn
    compile:
      opt_level: 0
      quantization: lite
      calibration:
        num_samples: 100
        method: ema
    runtime:
      device: 0
      async: true           # Default for dxnn profiles
      buffer_count: 6       # Engine I/O buffer count (1-100)
      use_ort: true         # Run unsupported ops on ONNX Runtime

  # NPU Quantize-Lite
  q-lite:
    target: dxnn
    compile:
      quantization: lite
      calibration:
        num_samples: 100
        method: ema
    runtime:
      device: 0

  # NPU Quantize-Pro (automatic Q-PRO pipeline)
  q-pro:
    target: dxnn
    compile:
      quantization:
        pro:
          num_samples: 1024
    runtime:
      device: 0
      async: true
```

> **Note:** `pro` enables dx_com's automatic Q-PRO pipeline
> (`use_q_pro=True`), which applies SmoothQuant (P0), QSNR-guided
> calibration refinement (P1), and per-layer AdaRound/FlexRound
> selection (P3/P4) automatically. It is mutually exclusive with the
> manual `p0`–`p5` enhanced-scheme keys and with `lite`/`master`.

#### Profile Naming Convention

| Profile | Target | Description |
|---------|--------|-------------|
| `onnx` | onnx | ONNX Runtime inference |
| `c-lite` | dxnn | Compile-Lite (opt_level=0, quantization=lite) |
| `q-lite` | dxnn | Quantize-Lite (quantization=lite) |
| `q-pro` | dxnn | Quantize-Pro (automatic Q-PRO pipeline) |

#### Quantization Options

| Value | Description |
|-------|-------------|
| `lite` | Lightweight quantization |
| `master` | Highest-accuracy quantization |
| `pro` | Automatic Q-PRO pipeline (`use_q_pro`; P0/P1/P3/P4 auto-applied) |
| `p0` – `p5` | Manual pro-level quantization (per-level parameter tuning) |
| `[p2, p3]` | Run multiple P-levels simultaneously |

#### Runtime Options

| Key | Type | Description |
|-----|------|-------------|
| `runtime.device` | `str \| int \| list` | Execution device. ONNX typically uses `cpu`/`gpu`; DXNN uses NPU device IDs. |
| `runtime.batch_size` | `int` | Evaluation batch size. Adjustable for ONNX; DXNN evaluation is effectively batch size 1. |
| `runtime.async` | `bool` | Enable async inference. Defaults to `false` for ONNX profiles and `true` for DXNN profiles. |
| `runtime.buffer_count` | `int` | DXNN-only engine I/O buffer count. Must be in `[1, 100]`. |
| `runtime.use_ort` | `bool` | DXNN-only fallback for unsupported ops via ONNX Runtime. Defaults to `true`. |

#### `compile.ppu_config`

`ppu_config` is an optional DXNN-only compile block for hardware postprocessing.

```yaml
compile:
  ppu_config:
    type: 0|1|2         # 0=anchor-based, 1=anchor-free single-head, 2=anchor-free multi-head
    num_classes: 80
    conf_thres: 0.5     # type 0,1 only
    activation: sigmoid # type 0 only
    topk: 100           # type 2 only
    layer: ...          # dict for type 0, list for type 1,2
```

| Key | Required For | Description |
|-----|---------------|-------------|
| `type` | all | PPU mode: `0` anchor-based, `1` anchor-free single-head, `2` anchor-free multi-head |
| `num_classes` | all | Number of classes |
| `conf_thres` | type `0`, `1` | Confidence threshold |
| `activation` | type `0` | Activation name such as `sigmoid` |
| `topk` | type `2` | Top-k candidate count |
| `layer` | all | Per-layer PPU description (`dict` for type `0`, `list` for type `1`/`2`) |

## Environment Variable Substitution

YAML files can reference environment variables using the `${VAR_NAME}` syntax:

```yaml
dataset:
  eval_path: ${DATA_ROOT}/ILSVRC2012/val  # Substitutes DATA_ROOT env var

artifacts:
  path: ${MODEL_ROOT}/${MODEL_NAME}       # MODEL_NAME equals the name field
```

Special variables:

| Variable | Description |
|----------|-------------|
| `${DATA_ROOT}` | Dataset root directory |
| `${MODEL_ROOT}` | Model artifact root directory |
| `${MODEL_NAME}` | Value of the YAML `name` field |
