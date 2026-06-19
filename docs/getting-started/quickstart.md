# Quick Start

## 1. Browse Available Models

```bash
# Interactive TUI browser
dxmz list

# Text output (for CI/scripts)
dxmz list --all

# Filter by domain/task
dxmz list --domain cv --task classification
```

## 2. View Model Information

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

## 3. Evaluate a Model

### Evaluate with ONNX Runtime

```bash
# Basic evaluation
dxmz eval resnet50_224x224 --profile onnx

# Specify data path
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

### Evaluate with NPU Runtime

```bash
# Evaluate a compiled dxnn model
dxmz eval resnet50_224x224 --profile q-lite
```

!!! info "NPU profiles are forced to batch_size=1"
    Profiles with `target: dxnn` always use `batch_size=1`, regardless of what is set in the YAML.
    Only ONNX profiles support adjustable `batch_size`.

## 4. Compile a Model

```bash
# Quantized compilation for NPU
dxmz compile resnet50_224x224 --profile q-lite

# Specify output directory
dxmz compile resnet50_224x224 --profile q-lite --output ./compiled/

# GPU-accelerated quantization
dxmz compile resnet50_224x224 --profile q-pro --use-gpu
```

## 5. Benchmark

```bash
# Benchmark all models with a profile
dxmz benchmark --profile onnx

# Filter by domain
dxmz benchmark --profile onnx --domain cv

# Save results
dxmz benchmark --profile onnx --output results.json
```

## 6. Run with a YAML File Directly

You can pass a YAML file path instead of a model name:

```bash
# Evaluate with a custom YAML
dxmz eval ./my_models/custom_model.yaml --profile onnx

# Compile with a custom YAML
dxmz compile ./my_models/custom_model.yaml --profile q-lite
```

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `DATA_ROOT` | Dataset root path | `/data/datasets` |
| `MODEL_ROOT` | Model artifact root path | `/data/models` |
| `DXMZ_MODEL_URL` | Auto-download server URL | `https://models.deepx.ai/v1` |

Settings can be placed in a `.env` file for automatic loading:

```bash title=".env"
DATA_ROOT=/data/datasets
MODEL_ROOT=/data/models
DXMZ_MODEL_URL=https://models.deepx.ai/v1
```
