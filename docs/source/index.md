# DX-ModelZoo

A unified framework for managing model preprocessing, postprocessing, evaluation, and NPU compilation — all driven by a single YAML configuration file.

## Key Features

<div class="grid cards" markdown>

-   📄 **YAML-Centric Configuration**

    ---

    Define entire model pipelines in a single YAML file — no Python code required. Preprocessing, postprocessing, and runtime profiles are all declared in one place.

-   🔌 **Extensible Registry System**

    ---

    Add custom preprocessing, postprocessing, datasets, and evaluators instantly using the `@REGISTRY.register()` decorator.

-   � **Auto-Discovery of Custom Ops**

    ---

    Place a `custom_ops.py` in your model directory; it's automatically imported and registered.

-   🚀 **Multi-Profile Execution**

    ---

    Manage multiple runtime profiles — ONNX, NPU/dxnn, and more — within a single YAML configuration.

-   🎯 **300+ Built-in Models**

    ---

    Production-ready CV models spanning classification, detection, segmentation, and beyond.

</div>

## Prerequisites

**Requirements**

- **Python**: 3.8+
- **DeepX Components**: DX-RT ≥ 3.0.0, DX-Compiler ≥ 2.4.0

## Quick Start

```bash
# Install DeepX Components
pip install dx_engine-*.whl dx_com-*.whl

# Clone and install DX-ModelZoo
git clone https://github.com/DEEPX-AI/dx-modelzoo.git
cd dx-modelzoo
pip install -e ".[cpu]"

# 1. Browse available models
dxmz list

# 2. View model information
dxmz info resnet50_224x224

# 3. Evaluate with ONNX Runtime
dxmz eval resnet50_224x224 --profile onnx

# 4. (Optional) Compile for NPU
dxmz compile resnet50_224x224 --profile q-lite
```

See [Quick Start Guide](getting-started/quickstart.md) for step-by-step instructions and [Installation Guide](getting-started/installation.md) for detailed setup.

<!-- BEGIN:SUPPORTED_TASKS -->
<!-- This section is auto-generated. Do not edit manually. -->
## Supported Tasks

| Category | Task | Models |
|---|---|---:|
| **Classification** | Image Classification | 102 |
| **Object Detection** | Object Detection | 112 |
|  | Oriented Object Detection | 5 |
| **Segmentation** | Semantic Segmentation | 8 |
|  | Instance Segmentation | 14 |
| **Pose / Landmark** | Pose Estimation | 10 |
|  | 3D Face Landmark | 2 |
|  | Hand Landmark | 1 |
| **Face** | Face Detection | 17 |
|  | Face Recognition | 4 |
|  | Face Attribute | 1 |
| **Person** | Pedestrian Attribute | 2 |
| **Image Restoration** | Image Denoising | 6 |
|  | Super Resolution | 3 |
|  | Low-Light Enhancement | 1 |
| **Depth** | Depth Estimation | 2 |
| **Other** | Zero Shot Instance Segmentation | 1 |

> **300+ models** across 17 CV tasks — all configurable via YAML.
<!-- END:SUPPORTED_TASKS -->

## Documentation Guide

| Section | Description |
|---|---|
| [Quick Start](getting-started/quickstart.md) | Quick introduction and step-by-step usage |
| [Installation](getting-started/installation.md) | Detailed installation and environment setup |
| [Guides](guides/yaml-config.md) | YAML config, evaluation, datasets, custom model authoring |
| [Architecture](architecture/overview.md) | Internal design, Registry pattern, pipeline flow |
| [API Reference](api/cli.md) | CLI, ModelBuilder, Pre/Post-processing, Evaluator, etc. |
