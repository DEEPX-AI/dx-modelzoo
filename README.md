# DX-ModelZoo

**YAML-Driven Model Management · Evaluation · NPU Compilation**

DX-ModelZoo is a unified framework for managing model preprocessing, postprocessing, evaluation, and NPU compilation — all driven by a single YAML configuration file.

## Key Features

- **YAML-Centric Configuration** — Define entire model pipelines in a single YAML file. No Python code required.
- **Extensible Registry System** — Add custom preprocessing, postprocessing, datasets, and evaluators via `@REGISTRY.register()`.
- **Auto-Discovery of Custom Ops** — Place a `custom_ops.py` in your model directory; it's automatically imported and registered.
- **Multi-Profile Execution** — Manage ONNX, NPU/dxnn, and other runtime profiles within one YAML.
- **300+ Built-in Models** — Production-ready CV and NLP models spanning classification, detection, segmentation, and beyond.

## Prerequisites

- **OS**: Ubuntu 20.04 LTS or 22.04 LTS
- **Python**: 3.8+
- **DeepX Components**:
  - [DeepX Runtime (DX-RT)](https://github.com/DEEPX-AI/dx-rt) ≥ 2.7.0 (compiled with `USE_ORT=ON`)
  - [DeepX Compiler (DX-COM)](https://github.com/DEEPX-AI/dx-com) ≥ 1.45.0
- **GPU** (optional):
  - NVIDIA GPU (Pascal/Turing/Ampere architecture)
  - NVIDIA Driver ≥ 535.230.02, CUDA 12.2, cuDNN 9.1

> Refer to the official DX-RT and DX-COM installation guides for detailed setup instructions.

## Installation

### 1. Install DeepX Components

```bash
# Install DX-RT python package
pip install dx_engine-*.whl

# Install DX-COM python package
pip install dx_com-*.whl
```

### 2. Install DX-ModelZoo

```bash
git clone https://github.com/DEEPX-AI/dx-modelzoo.git
cd dx-modelzoo
pip install -e ".[cpu]"
```

For GPU inference:

```bash
pip install -e ".[gpu]"
```

### Optional Extras

| Extra | Purpose |
|-------|---------|
| `cpu` | CPU-based ONNX inference |
| `gpu` | GPU-based ONNX inference |
| `dev` | Testing & linting (`pytest`, `ruff`) |
| `docs` | Documentation build (`mkdocs`) |
| `all` | CPU + dev + docs |

## Quick Start

```bash
# List available models (interactive TUI)
dxmz list

# View model details
dxmz info ResNet18

# Evaluate with ONNX Runtime
dxmz eval ResNet18 --profile onnx

# Compile for NPU
dxmz compile ResNet18 --profile q-lite

# Benchmark all models
dxmz benchmark --profile onnx --domain cv

# Scaffold a custom model (interactive wizard → ./custom)
dxmz create
```

### Evaluation Options

```bash
dxmz eval ResNet18 --profile onnx \
  --data-root /data/datasets \
  --model-root /data/models \
  --model-path /path/to/model.onnx \
  --seed 42 \    # Random seed for reproducible evaluation
  --save         # Save results to result/ as JSON
```

### Compile & Benchmark Options

```bash
# Compile with GPU-accelerated quantization
dxmz compile ResNet18 --profile q-lite --use-gpu --output ./out

# Benchmark across multiple NPU devices in parallel
dxmz benchmark --profile q-lite --devices 0,1,2,3 --save
```

### Using a YAML File Directly

```bash
dxmz eval ./my_models/custom_model.yaml --profile onnx
dxmz compile ./my_models/custom_model.yaml --profile q-lite
```

## YAML Configuration

Every model is defined by a single YAML file containing preprocessing, postprocessing, evaluation, and compilation settings. See the [YAML Configuration Guide](docs/guides/yaml-config.md) for the full specification.

```yaml
name: ResNet18
task: image_classification
inputs:
  - name: input
    shape: [1, 3, 224, 224]
    dtype: float32
preprocessing:
  - type: resize
    size: [256, 256]
  - type: centercrop
    height: 224
    width: 224
  - type: div
    x: 255
  - type: normalize
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
  - type: transpose
    axis: [2, 0, 1]
  - type: expanddim
    axis: 0
postprocessing:
  - type: topk
    k: [1, 5]
dataset:
  type: ILSVRC2012
  eval_path: ${DATA_ROOT}/ILSVRC2012/val
profiles:
  onnx:
    target: onnx
    runtime: { device: gpu, batch_size: 1 }
  q-lite:
    target: dxnn
    compile: { quantization: lite }
    runtime: { device: 0 }
```

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `DATA_ROOT` | Dataset root path | `download/datasets` |
| `MODEL_ROOT` | Model artifact root path | `download/models` |
| `DXMZ_MODEL_URL` | Auto-download server URL | `https://sdk.deepx.ai/modelzoo` |

## Documentation

For detailed guides, please refer to the full documentation:

| Topic | Link |
|-------|------|
| Installation | [docs/getting-started/installation.md](docs/getting-started/installation.md) |
| Quick Start | [docs/getting-started/quickstart.md](docs/getting-started/quickstart.md) |
| YAML Configuration | [docs/guides/yaml-config.md](docs/guides/yaml-config.md) |
| Model Evaluation | [docs/guides/evaluation.md](docs/guides/evaluation.md) |
| Custom Models | [docs/guides/custom-models.md](docs/guides/custom-models.md) |
| Architecture | [docs/architecture/overview.md](docs/architecture/overview.md) |
| Registry & Custom Ops | [docs/architecture/registry.md](docs/architecture/registry.md) |


## License

See [LICENSE](LICENSE) for details.
