# DX-ModelZoo

A unified framework for managing model preprocessing, postprocessing, evaluation, and NPU compilation — all driven by a single YAML configuration file.

## Key Features

- **YAML-Centric Configuration**: Define entire model pipelines in a single YAML file — no Python code required. Preprocessing, postprocessing, and runtime profiles are all declared in one place.  
- **Extensible Registry System**: Add custom preprocessing, postprocessing, datasets, and evaluators instantly using the `@REGISTRY.register()` decorator.  
- **Auto-Discovery of Custom Ops**: Place a `custom_ops.py` in your model directory; it's automatically imported and registered.  
- **Multi-Profile Execution**: Manage multiple runtime profiles — ONNX, NPU/dxnn, and more — within a single YAML configuration.  
- **300+ Built-in Models**: Production-ready CV models spanning classification, detection, segmentation, and beyond.  

## Prerequisites

See [Installation](02_01_Installation.md#prerequisites) for the required Python version and DeepX Components.

## Get Started

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

See [Quick Start Guide](00_Quick_Start.md) for step-by-step instructions and [Installation Guide](02_01_Installation.md) for detailed setup.

<!-- BEGIN:SUPPORTED_TASKS -->
<!-- Counts are YAML variant counts under src/dx_modelzoo/models/cv. -->
## Supported Tasks

| Category | Task | Model Variants |
|---|---|---:|
| **Classification** | Image Classification | 137 |
|  | Zero-Shot Image Classification | 6 |
|  | Image Retrieval | 1 |
| **Object Detection** | Object Detection | 124 |
|  | Face Detection | 18 |
|  | Oriented Object Detection | 5 |
|  | Mainbody Detection | 2 |
|  | Hand Detection | 1 |
|  | 3D Object Detection | 1 |
| **Segmentation** | Instance Segmentation | 35 |
|  | Semantic Segmentation | 24 |
|  | Zero-Shot Instance Segmentation | 1 |
|  | Panoptic Driving Perception | 1 |
| **Pose / Landmark** | Pose Estimation | 21 |
|  | Face Landmark | 2 |
|  | Hand Landmark | 1 |
|  | Keypoint Detection | 1 |
|  | Object Pose Estimation | 1 |
| **Face / Person** | Face Recognition | 4 |
|  | Person Attribute | 2 |
|  | Face Attribute | 1 |
|  | Person Re-Identification | 1 |
| **Image Restoration** | Super Resolution | 6 |
|  | Image Denoising | 5 |
|  | Low-Light Enhancement | 2 |
|  | Image Matting | 2 |
| **Other** | Depth Estimation | 10 |
|  | Anomaly Detection | 4 |
|  | Visual Place Recognition | 3 |

> **422 model variants** across **29 CV tasks** — all configurable via YAML.
>
> Counts are YAML definitions (variants) under `src/dx_modelzoo/models/cv`, not
> distinct architectures. A single architecture may ship several variants
> (different input sizes, decode options, or PPU / pre-optimized builds).
<!-- END:SUPPORTED_TASKS -->

## Documentation Guide

| Section | Description |
|---|---|
| [Quick Start](00_Quick_Start.md) | Quick introduction and step-by-step usage |
| [Installation](02_01_Installation.md) | Detailed installation and environment setup |
| [Guides](02_02_Model_Evaluation.md) | Model evaluation workflows and CLI usage |
| [Architecture](03_01_Architecture_Overview.md) | Internal design, Registry pattern, pipeline flow |
| [API Reference](04_01_CLI_Reference.md) | CLI, ModelBuilder, Pre/Post-processing, Evaluator, etc. |
