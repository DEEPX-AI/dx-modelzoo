# DX-ModelZoo

**YAML-Driven Model Management · Evaluation · NPU Compilation**

DX-ModelZoo is a unified framework for managing model preprocessing, postprocessing, evaluation, and NPU compilation — all driven by a single YAML configuration file.

## Key Features

- **YAML-Centric Configuration** — Define entire model pipelines in a single YAML file. No Python code required.
- **Extensible Registry System** — Add custom preprocessing, postprocessing, datasets, and evaluators via `@REGISTRY.register()`.
- **Auto-Discovery of Custom Ops** — Place a `custom_ops.py` in your model directory; it's automatically imported and registered.
- **Multi-Profile Execution** — Manage ONNX, NPU/dxnn, and other runtime profiles within one YAML.
- **300+ Built-in Models** — Production-ready CV models spanning classification, detection, segmentation, and beyond.

## Prerequisites

- **Python**: 3.8+
- **DeepX Components**:
  - [DeepX Runtime (DX-RT)](https://github.com/DEEPX-AI/dx-rt) ≥ 3.0.0 (compiled with `USE_ORT=ON`)
  - [DeepX Compiler (DX-Compiler)](https://github.com/DEEPX-AI/dx-compiler) ≥ 2.4.0
- **GPU** (optional): NVIDIA GPU with CUDA 12.2+ for GPU inference

> See the [Installation Guide](source/getting-started/installation.md) for detailed setup instructions including WSL2 and GPU configuration.

## Installation

### 1. Install DeepX Components

```bash
# Install DX-RT python package
pip install dx_engine-*.whl

# Install DX-Compiler python package
pip install dx_com-*.whl
```

### 2. Install DX-ModelZoo

```bash
git clone https://github.com/DEEPX-AI/dx-modelzoo.git
cd dx-modelzoo
pip install -e ".[cpu]"   # or ".[gpu]" for GPU inference
```

## Quick Start

### 1. Set up environment

```bash
# Point to your datasets location
export DATA_ROOT=/path/to/datasets

# (Optional) Point to model files location
export MODEL_ROOT=/path/to/models
```

### 2. Browse available models

```bash
dxmz list
```

### 3. View model information

```bash
dxmz info resnet50_224x224
```

### 4. Evaluate with ONNX Runtime

```bash
dxmz eval resnet50_224x224 --profile onnx
```

### 5. Compile for NPU

```bash
dxmz compile resnet50_224x224 --profile q-lite
```



## Configuration

Every model is defined by a single YAML file containing preprocessing, postprocessing, dataset, and runtime profiles.

**Available Profiles:**

- **onnx** — ONNX Runtime (CPU/GPU)
- **c-lite** — Compile only with lite quantization
- **q-lite** — Compile and run on NPU with lite quantization
- **q-pro** — Compile and run on NPU with pro quantization

**Environment Variables:** Use `${DATA_ROOT}`, `${MODEL_ROOT}`, and `${DXMZ_MODEL_URL}` in YAML files for path configuration.

See the [YAML Configuration Guide](docs/source/guides/yaml-config.md) for complete details and examples.

## Documentation

For detailed guides, please refer to the full documentation:

| Topic | Link |
|-------|------|
| Installation | [docs/source/getting-started/installation.md](docs/source/getting-started/installation.md) |
| Quick Start | [docs/source/getting-started/quickstart.md](docs/source/getting-started/quickstart.md) |
| YAML Configuration | [docs/source/guides/yaml-config.md](docs/source/guides/yaml-config.md) |
| Model Evaluation | [docs/source/guides/evaluation.md](docs/source/guides/evaluation.md) |
| Custom Models | [docs/source/guides/custom-models.md](docs/source/guides/custom-models.md) |
| Architecture | [docs/source/architecture/overview.md](docs/source/architecture/overview.md) |
| Registry & Custom Ops | [docs/source/architecture/registry.md](docs/source/architecture/registry.md) |


## License

See [LICENSE](LICENSE) for details.

---
