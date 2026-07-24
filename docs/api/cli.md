# CLI Reference

DX ModelZoo performs all operations through the `dxmz` CLI tool. It is built on [Typer](https://typer.tiangolo.com/).

## Verify Installation

```bash
dxmz --help
```

## Command Overview

| Command | Description |
|---------|-------------|
| `eval` | Model evaluation |
| `compile` | NPU compilation |
| `benchmark` | Multi-model benchmarking |
| `list` | List available models |
| `info` | Model details |
| `create` | Create a new model YAML interactively |

---

## `dxmz eval`

Evaluates one or more models on their datasets and outputs metrics.

```bash
dxmz eval <model> [<model2> ...] --profile <profile> [options]
```

### Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `model` | One or more model names or YAML paths | `resnet50_224x224`, `./my_model.yaml` |

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--profile` | Runtime profile | (required) |
| `--data-root` | Dataset root directory (overrides `DATA_ROOT`) | Environment variable |
| `--model-root` | Model file root directory (overrides `MODEL_ROOT`) | Environment variable |
| `--model-path` | Explicit path to the model file | — |
| `--save` | Save results as JSON | `False` |
| `--seed` | Random seed for reproducible evaluation | `42` |

### Examples

```bash
# Basic evaluation
dxmz eval resnet50_224x224 --profile onnx

# Multiple models
dxmz eval resnet50_224x224 mobilenetv3-small_224x224 --profile onnx

# Specify data/model paths
dxmz eval resnet50_224x224 --profile onnx \
  --data-root /data/datasets \
  --model-root /data/models

# Explicit model file path + save results
dxmz eval resnet50_224x224 --profile q-lite \
  --model-path /path/to/resnet50.dxnn \
  --save

# Pipeline model with stage-specific model paths
dxmz eval ocr_pipeline.yaml --profile onnx \
  --model-path "det=/path/det.onnx,rec=/path/rec.onnx"
```

---

## `dxmz compile`

Compiles a model for the NPU.

```bash
dxmz compile <model> --profile <profile> [options]
```

### Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `model` | Model name or YAML path | `yolov8n_640x640` |

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--profile` | Compilation profile (`c-lite`, `q-lite`, `q-pro`) | (required) |
| `--output` | Output directory or file path | Auto-generated |
| `--model-path` | Source ONNX model path | — |
| `--data-root` | Calibration data path | Environment variable |
| `--model-root` | Model file root | Environment variable |
| `--use-gpu` | Use GPU for quantization calibration | `False` |

### Examples

```bash
# Basic compilation
dxmz compile resnet50_224x224 --profile q-lite

# Specify output directory
dxmz compile resnet50_224x224 --profile q-pro --output ./compiled/

# GPU-accelerated compilation
dxmz compile yolov8n_640x640 --profile q-lite --use-gpu

# Pipeline model with stage paths
dxmz compile ocr_pipeline.yaml --profile q-lite \
  --model-path "det=/path/det.onnx,rec=/path/rec.onnx"
```

---

## `dxmz benchmark`

Benchmarks multiple models in batch with optional parallel execution.

```bash
dxmz benchmark --profile <profile> [options]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--profile` | Runtime profile | (required) |
| `--models-dir` | Directory containing model YAMLs | Builtin models |
| `--data-root` | Dataset root | Environment variable |
| `--model-root` | Model file root | Environment variable |
| `--domain` | Domain filter (e.g., `cv`) | All |
| `--task` | Task filter (e.g., `image_classification`) | All |
| `--devices` | Comma-separated device IDs for parallel execution | — |
| `--save` | Save results to JSON file | `False` |

### Examples

```bash
# Benchmark all models
dxmz benchmark --profile onnx

# Benchmark specific task
dxmz benchmark --profile onnx --task image_classification

# Parallel on multiple NPU devices
dxmz benchmark --profile q-lite --devices 0,1,2,3

# Save results
dxmz benchmark --profile q-lite --save
```

---

## `dxmz list`

Displays the list of available models. By default, an interactive tree browser is launched.

```bash
dxmz list [options]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--domain` | Domain filter | All |
| `--task` | Task filter | All |
| `--all`, `-a` | Output as plain text instead of the interactive UI | `False` |

### Examples

```bash
# Interactive model browser
dxmz list

# Full list in plain text
dxmz list --all

# Filter by specific task
dxmz list -a --task object_detection
```

!!! tip "Interactive Browser"
    Running `dxmz list` without any options launches a TUI (Text User Interface) tree browser. Use the arrow keys to navigate and press Enter to view model details.

---

## `dxmz info`

Displays detailed information about a specific model.

```bash
dxmz info <model>
```

### Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `model` | Model name or YAML path | `resnet50_224x224` |

### Examples

```bash
dxmz info resnet50_224x224
```

Sample output:

```
Name: resnet50_224x224
Inputs: [{'name': 'input', 'shape': [1, 3, 224, 224], 'dtype': 'float32'}]
Profiles: ['onnx', 'c-lite', 'q-lite', 'q-pro']
Dataset: {'type': 'ILSVRC2012', 'eval_path': '${DATA_ROOT}/ILSVRC2012/val'}
```

---

## `dxmz create`

Interactively create a new model YAML configuration file.

```bash
dxmz create
```

Launches a step-by-step wizard that asks for model name, task, input shape, preprocessing, and generates the YAML file.

---

## Environment Variables

Key environment variables used by the CLI:

| Variable | Description | CLI Override |
|----------|-------------|-------------|
| `DATA_ROOT` | Dataset root directory | `--data-root` |
| `MODEL_ROOT` | Model file root directory | `--model-root` |
| `DXMZ_MODEL_URL` | Model auto-download base URL | — |
| `DXNN_DEVICES` | DxRuntime device specification | — |

!!! warning "Environment Variable Precedence"
    When CLI options (`--data-root`, `--model-root`) are specified, they take precedence over environment variables.

## Profile Types

| Profile | Target | Description |
|---------|--------|-------------|
| `onnx` | ONNX Runtime | CPU/GPU inference |
| `c-lite` | dxnn | Compile-Lite (opt_level=0, minimal optimization) |
| `q-lite` | dxnn | Quantize-Lite (EMA calibration) |
| `q-pro` | dxnn | Quantize-Pro (higher accuracy quantization) |
