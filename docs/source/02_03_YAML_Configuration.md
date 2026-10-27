# YAML Model Configuration

Every model in DX-ModelZoo is defined by a single YAML file. No Python code is needed — preprocessing, postprocessing, evaluation, and compilation settings are all included.

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

dataset:                      # Dataset configuration
  type: ILSVRC2012
  eval_path: ${DATA_ROOT}/ILSVRC2012/val

evaluator:                    # Evaluator type
  type: image_classification

artifacts:                    # Model file paths
  path: ${MODEL_ROOT}/${MODEL_NAME}

calibration: &calibration
  method: ema
  num: 100
  seed: 42

profiles:                     # Runtime profiles
  onnx:
    target: onnx
    runtime:
      device: gpu
      batch_size: 1
  q-lite:
    target: dxnn
    compile:
      calibration: *calibration
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

### `reference` / `description`

Optional metadata fields for documentation:

```yaml
reference: https://arxiv.org/abs/1512.03385  # Paper or GitHub URL
description: |
  ResNet-50 trained on ImageNet-1K.
  Top-1 accuracy: 76.13%
  Top-5 accuracy: 93.15%
```

- **reference**: URL to the paper, GitHub repo, or model card
- **description**: Multiline string describing the model, training details, and performance

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

!!! note "NPU arithmetic behavior"  
    For a `target: dxnn` profile, arithmetic operations such as `div`,
    `normalize`, `transpose`, and `expanddim` are skipped when runtime input
    dtype is `uint8`. They remain for `float32` input. Set
    `appendix.npu_skip_arithmetic: false` when a model requires CPU-side
    arithmetic preprocessing.

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
    conf_thres: 0.25
    iou_thres: 0.45
    max_output_boxes: 300

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
| `COCO` | Detection | `COCO` |
| `PascalVOC2007` | Detection | `VOCdevkit/VOC2007` |
| `Cityscapes` | Segmentation | `cityscapes` |
| `ADE20K` | Segmentation | `ADEChallengeData2016` |
| `WiderFace` | Face detection | `WIDER_FACE` |
| `COCOPose` | Pose estimation | `COCO` |
| `NYUDepthv2` | Depth estimation | `NYU_Depth_V2` |
| `BSD68` | Denoising | `BSD68` |
| `BSD100` | Super resolution | `BSD100/bicubic_4x` |
| `DOTAv1` | Oriented detection | `DOTA/val` |
| `LFW` | Face verification | `lfw` |

### `evaluator`

Specifies the evaluator type for computing metrics.

```yaml
evaluator:
  type: image_classification
```

The evaluator type typically matches the task type. Common evaluators:

| Evaluator | Metrics |
|-----------|----------|
| `image_classification` | Top-1, Top-5 Accuracy |
| `object_detection` | mAP@0.5, mAP@0.5:0.95 |
| `semantic_segmentation` | mIoU, Pixel Accuracy |
| `instance_segmentation` | bbox mAP, segm mAP |
| `face_detection` | mAP, Recall |
| `pose_estimation` | AP, AR |

### `artifacts`

Defines where to find model files (ONNX, weights, etc.).

```yaml
artifacts:
  path: ${MODEL_ROOT}/${MODEL_NAME}
```

- **path**: Directory containing model files
- Uses `${MODEL_ROOT}` environment variable + `${MODEL_NAME}` (auto-set to the `name` field)
- DX-ModelZoo automatically downloads models if not found locally

### `profiles`

Defines multiple runtime profiles. Each profile specifies its own target, compilation options, and runtime options.

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
      calibration: *calibration
    runtime:
      device: 0
      async: true           # Default for dxnn profiles
      buffer_count: 6       # Engine I/O buffer count (1-100)
      use_ort: true         # Run unsupported ops on ONNX Runtime

  # NPU Quantize-Lite
  q-lite:
    target: dxnn
    compile:
      calibration: *calibration
    runtime:
      device: 0

  # NPU Quantize-Pro (automatic Q-PRO pipeline)
  q-pro:
    target: dxnn
    compile:
      calibration:
        <<: *calibration
        num: 1024
        use_q_pro: true
    runtime:
      device: 0
      async: true
```

!!! note "NOTE"  
    `calibration.use_q_pro: true` enables dx_com's automatic Q-PRO pipeline.
    Calibration settings are passed through the profile's `compile` block.

#### Profile Naming Convention

| Profile | Target | Description |
|---------|--------|-------------|
| `onnx` | onnx | ONNX Runtime inference |
| `c-lite` | dxnn | Compile profile with `opt_level: 0` |
| `q-lite` | dxnn | NPU profile with standard calibration |
| `q-pro` | dxnn | NPU profile with `calibration.use_q_pro: true` |

#### Calibration Options

These keys are typically defined once in a top-level `calibration:` block (using a YAML anchor, e.g. `calibration: &calibration`) and reused across profiles via `calibration: *calibration`, as shown in [Full Structure](#full-structure).

| Key | Description |
|-------|-------------|
| `compile.calibration.method` | Calibration method, such as `ema` |
| `compile.calibration.num` | Number of calibration samples |
| `compile.calibration.seed` | Calibration random seed |
| `compile.calibration.use_q_pro` | Enable automatic Q-PRO calibration |

#### Runtime Options

| Key | Type | Description |
|-----|------|-------------|
| `runtime.device` | `str \| int \| list` | Execution device. ONNX typically uses `cpu`/`gpu`; DXNN uses NPU device IDs. |
| `runtime.batch_size` | `int` | Evaluation batch size. Adjustable for ONNX; DXNN evaluation is effectively batch size 1. |
| `runtime.async` | `bool` | Enable async inference. Defaults to `false` for ONNX profiles and `true` for DXNN profiles. |
| `runtime.buffer_count` | `int` | DXNN-only engine I/O buffer count. Must be in `[1, 100]`. |
| `runtime.use_ort` | `bool` | DXNN-only fallback for unsupported ops via ONNX Runtime. Defaults to `true`. |

#### `compile.ppu_config`

`ppu_config` is an optional DXNN-only compile block for postprocessing. Types `0` and `1` use PPU hardware; type `2` is a CPU-side TopK optimization for DFL-based YOLO models and does not use PPU hardware.

```yaml
compile:
  ppu_config:
    type: 0|1|2         # 0=anchor-based, 1=anchor-free single-head, 2=DFL-based CPU TopK (deprecated)
    num_classes: 80
    conf_thres: 0.5     # type 0,1 only
    activation: sigmoid # type 0 only
    topk: 100           # type 2 only
    layer: ...          # dict for type 0, list for type 1,2
```

| Key | Required For | Description |
|-----|---------------|-------------|
| `type` | all | Mode: `0` anchor-based PPU, `1` anchor-free single-head PPU, `2` DFL-based CPU TopK (no PPU hardware) |
| `num_classes` | all | Number of classes |
| `conf_thres` | type `0`, `1` | Confidence threshold |
| `activation` | type `0` | Activation name such as `sigmoid` |
| `topk` | type `2` | Top-k candidate count |
| `layer` | all | Per-layer PPU description (`dict` for type `0`, `list` for type `1`/`2`) |

!!! warning "PPU type 2 is deprecated"
    DX-Compiler deprecated type `2` in v2.4.0. Use [`dx_com.pre_optimize()`](https://github.com/DEEPX-AI/dx-compiler/blob/main/source/docs/02_09_Pre_Optimize_API.md) for the recommended TopK-first optimization instead.

## Environment Variable Substitution

YAML files can reference environment variables using the `${VAR_NAME}` syntax:

```yaml
dataset:
  eval_path: ${DATA_ROOT}/ILSVRC2012/val  # Substitutes DATA_ROOT env var

artifacts:
  path: ${MODEL_ROOT}/${MODEL_NAME}       # MODEL_NAME equals the name field
```

Special variables:

| Variable | Description | Example |
|----------|-------------|----------|
| `${MODEL_NAME}` | Value of the YAML `name` field (not an OS environment variable) | `resnet50_224x224` |

`${DATA_ROOT}`, `${MODEL_ROOT}`, and `${DXMZ_MODEL_URL}` follow the same `${VAR_NAME}` substitution syntax — see [CLI Reference](04_01_CLI_Reference.md#environment-variables) for the full list and how to set them.

`DXMZ_MODEL_URL` is used by the session factory to construct download URLs:

- ONNX: `${DXMZ_MODEL_URL}/onnx/<model_name>.onnx`
- DXNN: `${DXMZ_MODEL_URL}/<profile>-dxnn/latest/<model_name>_<profile>.dxnn`

These values can be set via:  

- Environment variables: `export DATA_ROOT=/data/datasets`
- `.env` file in the project root
- Command-line options: `dxmz eval ... --data-root /data/datasets`
