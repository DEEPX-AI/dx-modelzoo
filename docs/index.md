# DX ModelZoo

**YAML-Driven Model Management · Evaluation · NPU Compilation**

---

DX ModelZoo is a unified framework for managing model preprocessing, postprocessing, evaluation, and NPU compilation — all driven by a single YAML configuration file. Define once, run everywhere.

## Key Features

<div class="grid cards" markdown>

-   📄 **YAML-Centric Configuration**

    ---

    Define entire model pipelines in a single YAML file — no Python code required. Preprocessing, postprocessing, and runtime profiles are all declared in one place.

-   🔌 **Extensible Registry System**

    ---

    Add custom preprocessing, postprocessing, datasets, and evaluators instantly using the `@REGISTRY.register()` decorator.

-   🚀 **Multi-Profile Execution**

    ---

    Manage multiple runtime profiles — ONNX, NPU/dxnn, and more — within a single YAML configuration.

-   🎯 **Extensive Built-in Models**

    ---

    Production-ready CV models spanning classification, detection, segmentation, and beyond.

</div>

## Quick Start

```bash
# Install
pip install -e ".[all]"

# List available models
dxmz list

# Evaluate with ONNX Runtime
dxmz eval resnet50_224x224 --profile onnx

# Compile for NPU
dxmz compile resnet50_224x224 --profile q-lite
```

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

> **291+ models** across 17 CV tasks — all configurable via YAML.
<!-- END:SUPPORTED_TASKS -->

## Documentation Guide

| Section | Description |
|---|---|
| [Getting Started](getting-started/installation.md) | Installation and first evaluation run |
| [Guides](guides/yaml-config.md) | YAML config, evaluation, datasets, custom model authoring |
| [Architecture](architecture/overview.md) | Internal design, Registry pattern, pipeline flow |
| [API Reference](api/cli.md) | CLI, ModelBuilder, Pre/Post-processing, Evaluator, etc. |
