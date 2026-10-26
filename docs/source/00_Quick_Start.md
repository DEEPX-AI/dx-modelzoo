# Quick Start

Quick walkthrough of the core DX-ModelZoo workflow: browse, evaluate, and compile models in minutes.

!!! note "Prerequisites"  
    Before running these commands, make sure you have [installed DX-ModelZoo](02_01_Installation.md).

## Quick Start Overview

**Basic Workflow**

- Step 1. [Browse Available Models](#step-1-browse-available-models) - Explore pre-configured models
- Step 2. [View Model Information](#step-2-view-model-information) - Check model details and profiles
- Step 3. [Evaluate with ONNX Runtime](#step-3-evaluate-with-onnx-runtime) - Run CPU/GPU inference
- Step 4. [Evaluate with NPU Runtime](#step-4-evaluate-with-npu-runtime) - Run NPU inference (optional)

**Advanced Options (choose as needed)**

- Optional 1. [Compile a Model](#optional-1-compile-a-model) - Compile for NPU deployment
- Optional 2. [Benchmark](#optional-2-benchmark) - Batch evaluation across multiple models
- Optional 3. [Run with YAML File](#optional-3-run-with-a-yaml-file-directly) - Use custom configurations
- Optional 4. [Create a Custom Model](#optional-4-create-a-custom-model) - Add your own models

## Basic Workflow

### Step 1: Browse Available Models

```bash
# Interactive TUI browser
dxmz list

# Text output (for CI/scripts)
dxmz list --all

# Filter by domain/task
dxmz list --domain cv --task image_classification
```

### Step 2: View Model Information

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
    - **c-lite**: Compile profile with `opt_level: 0`
    - **q-lite**: Compile and run on NPU with standard calibration
    - **q-pro**: Compile and run on NPU with `use_q_pro: true`

### Step 3: Evaluate with ONNX Runtime

```bash
dxmz eval resnet50_224x224 --profile onnx --save
```

See [CLI Reference](04_01_CLI_Reference.md#dxmz-eval) for all options (`--data-root`, `--model-path`, `--seed`, etc.).

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

### Step 4: Evaluate with NPU Runtime

```bash
# Evaluate a compiled dxnn model
dxmz eval resnet50_224x224 --profile q-lite
```

!!! note "Compiled model availability"  
    DX-ModelZoo does not compile a missing model during evaluation. For a
    `target: dxnn` profile, provide the `.dxnn` artifact in the configured
    artifact directory or set `DXMZ_MODEL_URL` so the matching artifact can be
    downloaded. Use `dxmz compile` to create the artifact explicitly.

!!! note "NPU profiles are forced to batch_size=1"  
    Profiles with `target: dxnn` always use `batch_size=1`, regardless of what is set in the YAML.
    Only ONNX profiles support adjustable `batch_size`.

## Advanced Options (choose as needed)

### Optional 1: Compile a Model

```bash
dxmz compile resnet50_224x224 --profile q-lite
```

See [CLI Reference](04_01_CLI_Reference.md#dxmz-compile) for all options (`--output`, `--profile q-pro`, etc.).

### Optional 2: Benchmark

```bash
dxmz benchmark --profile onnx --save
```

See [CLI Reference](04_01_CLI_Reference.md#dxmz-benchmark) for domain/task filters and multi-device options (`--domain`, `--devices`, etc.).

Benchmark results are saved in JSON format:

```text
result/benchmark_<profile>_<timestamp>.json
```

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

### Optional 3: Run with a YAML File Directly

```bash
# Evaluate with a custom YAML
dxmz eval ./my_models/custom_model.yaml --profile onnx

# Compile with a custom YAML
dxmz compile ./my_models/custom_model.yaml --profile q-lite
```

### Optional 4: Create a Custom Model

```bash
# Scaffold a new custom model (interactive wizard)
dxmz create
```

This creates a template in the `./custom/` directory with:  

- YAML configuration file  
- Custom preprocessing/postprocessing stubs  
- Dataset loader template  

## Environment Variables

See [CLI Reference](04_01_CLI_Reference.md#environment-variables) for the full list of environment variables. The most common ones can be placed in a `.env` file for automatic loading:

```bash title=".env"
DATA_ROOT=/data/datasets
MODEL_ROOT=/data/models
DXMZ_MODEL_URL=https://sdk.deepx.ai/modelzoo
```
