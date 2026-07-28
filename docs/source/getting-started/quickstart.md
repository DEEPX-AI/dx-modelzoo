# Quick Start

Quick walkthrough of the core DX-ModelZoo workflow: browse, evaluate, and compile models in minutes.

!!! note "Prerequisites"
    Before running these commands, make sure you have [installed DX-ModelZoo](installation.md).

!!! note "See Also"
    - [Installation](installation.md) - Installation prerequisites and setup
    - [Model Evaluation](../guides/evaluation.md) - Detailed evaluation guide
    - [CLI Reference](../api/cli.md) - Complete CLI command reference

## Quick Start Overview

**Basic Workflow:**

1. [Browse Available Models](#step-1-browse-available-models) - Explore pre-configured models
2. [View Model Information](#step-2-view-model-information) - Check model details and profiles
3. [Evaluate with ONNX Runtime](#step-3-evaluate-with-onnx-runtime) - Run CPU/GPU inference
4. [Evaluate with NPU Runtime](#step-4-evaluate-with-npu-runtime) - Run NPU inference (optional)

**Advanced Options (choose as needed):**

- [Compile a Model](#optional-compile-a-model) - Compile for NPU deployment
- [Benchmark](#optional-benchmark) - Batch evaluation across multiple models
- [Run with YAML File](#optional-run-with-a-yaml-file-directly) - Use custom configurations
- [Create a Custom Model](#optional-create-a-custom-model) - Add your own models

---

## Step 1: Browse Available Models

```bash
# Interactive TUI browser
dxmz list

# Text output (for CI/scripts)
dxmz list --all

# Filter by domain/task
dxmz list --domain cv --task classification
```

## Step 2: View Model Information

```bash
dxmz info resnet50_224x224
```

Example output:
```
Name: resnet50_224x224
Inputs: [{'name': 'input', 'shape': [1, 3, 224, 224], 'dtype': 'float32', 'layout': 'NCHW'}]
Profiles: ['onnx', 'c-lite', 'q-lite', 'q-pro']
Dataset: {'type': 'ILSVRC2012', 'eval_path': '${DATA_ROOT}/ILSVRC2012/val'}
```

!!! note "Available Profiles"
    - **onnx**: ONNX Runtime (CPU/GPU)
    - **c-lite**: Compile-only (lite quantization)
    - **q-lite**: Compile & run on NPU (lite quantization)
    - **q-pro**: Compile & run on NPU (pro quantization)

## Step 3: Evaluate with ONNX Runtime

```bash
# Basic evaluation
dxmz eval resnet50_224x224 --profile onnx

# Specify data path (required if DATA_ROOT env var is not set)
dxmz eval resnet50_224x224 --profile onnx --data-root /data/datasets

# Specify a model file
dxmz eval resnet50_224x224 --profile onnx --model-path /models/resnet50_224x224.onnx

# Save results as JSON
dxmz eval resnet50_224x224 --profile onnx --save

# With custom seed
dxmz eval resnet50_224x224 --profile onnx --seed 123
```

With `--save`, a JSON file is saved to the `result/` directory:

```json
{
  "model": "resnet50_224x224",
  "dataset": "imagenet",
  "metrics": [
    {"name": "Top-1", "metric_value": 69.76},
    {"name": "Top-5", "metric_value": 89.08}
  ],
  "fps": 624.7,
  "elapsed_time": 82,
  "profile": "onnx"
}
```

## Step 4: Evaluate with NPU Runtime

```bash
# Evaluate a compiled dxnn model
dxmz eval resnet50_224x224 --profile q-lite
```

!!! note "Auto-compilation"
    If the compiled model doesn't exist, DX-ModelZoo will automatically compile it first.

!!! note "NPU profiles are forced to batch_size=1"
    Profiles with `target: dxnn` always use `batch_size=1`, regardless of what is set in the YAML.
    Only ONNX profiles support adjustable `batch_size`.

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `DATA_ROOT` | Dataset root path | `/data/datasets` |
| `MODEL_ROOT` | Model artifact root path | `/data/models` |
| `DXMZ_MODEL_URL` | Auto-download server URL | `https://sdk.deepx.ai/modelzoo` |

Settings can be placed in a `.env` file for automatic loading:

```bash title=".env"
DATA_ROOT=/data/datasets
MODEL_ROOT=/data/models
DXMZ_MODEL_URL=https://sdk.deepx.ai/modelzoo
```

## Optional: Compile a Model

```bash
# Quantized compilation for NPU
dxmz compile resnet50_224x224 --profile q-lite

# Specify output directory
dxmz compile resnet50_224x224 --profile q-lite --output ./compiled/

# GPU-accelerated quantization
dxmz compile resnet50_224x224 --profile q-pro --use-gpu
```

## Optional: Benchmark

```bash
# Benchmark all models with a profile
dxmz benchmark --profile onnx

# Filter by domain
dxmz benchmark --profile onnx --domain cv

# Benchmark across multiple NPU devices in parallel
dxmz benchmark --profile q-lite --devices 0,1,2,3

# Save results as JSON
dxmz benchmark --profile onnx --output results.json
```

Benchmark results are saved in JSON format:

```json
[
  {
    "model": "resnet50_224x224",
    "profile": "onnx",
    "metrics": [
      {"name": "Top-1", "metric_value": 69.76},
      {"name": "Top-5", "metric_value": 89.08}
    ],
    "fps": 624.7,
    "elapsed_time": 82
  }
]
```

## Optional: Run with a YAML File Directly

You can pass a YAML file path instead of a model name:

```bash
# Evaluate with a custom YAML
dxmz eval ./my_models/custom_model.yaml --profile onnx

# Compile with a custom YAML
dxmz compile ./my_models/custom_model.yaml --profile q-lite
```

## Optional: Create a Custom Model

```bash
# Scaffold a new custom model (interactive wizard)
dxmz create
```

This creates a template in the `./custom/` directory with:

- YAML configuration file
- Custom preprocessing/postprocessing stubs
- Dataset loader template
