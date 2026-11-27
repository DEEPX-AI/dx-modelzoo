<div align="center">

# DX-ModelZoo

**Evaluate, compile and manage models for DEEPX NPUs from a single YAML file.**

[![Release](https://img.shields.io/github/v/release/DEEPX-AI/dx-modelzoo)](https://github.com/DEEPX-AI/dx-modelzoo/releases)
[![Docs](https://img.shields.io/badge/docs-Tech%20Docs-blue)](https://developer.deepx.ai/tech-docs/DXNN-SDK/DX-ModelZoo/DX-ModelZoo_Overview/)

</div>

DX-ModelZoo (repo/package `dx-modelzoo`, import `dx_modelzoo`, CLI `dxmz`) manages model preprocessing,
postprocessing, evaluation and NPU compilation, all driven by one YAML file per model. It works with
[DX-COM](https://github.com/DEEPX-AI/dx-compiler) and [DX-RT](https://github.com/DEEPX-AI/dx_rt), which are
versioned together in [DX-AllSuite](https://github.com/DEEPX-AI/dx-all-suite).

- **Built-in model library** — CV models for classification, detection, segmentation and more, each defined in one YAML file.
- **One YAML, many profiles** — run the same model on ONNX Runtime (`onnx`) or the NPU (`c-lite`, `q-lite`, `q-pro`).
- **Extensible** — register custom preprocessing, postprocessing, datasets and evaluators with `@REGISTRY.register()`; a `custom_ops.py` in the model directory is imported automatically.

## What You Need

| | |
|---|---|
| **OS** | Linux (Windows via WSL2) |
| **Python** | 3.8–3.14 |
| **DX-RT** | v3.5.0+, built with `USE_ORT=ON` — only to run NPU profiles (`c-lite`, `q-lite`, `q-pro`) |
| **DX-COM** | v2.5.0+ — only for `dxmz compile` (x86_64 host only) |
| **GPU** (optional) | NVIDIA GPU with CUDA 12.2 for GPU inference |

The `onnx` profile needs neither DX-RT nor DX-COM.

## Quick Start

Install DX-ModelZoo and evaluate ResNet-50 on ONNX Runtime (prints Top-1/Top-5 accuracy):

```bash
git clone https://github.com/DEEPX-AI/dx-modelzoo.git
cd dx-modelzoo
pip install -e ".[cpu]"                        # or ".[gpu]"; install only one of them

export DATA_ROOT=/path/to/datasets             # dataset layout → Datasets Guide
dxmz list                                      # browse the built-in models
dxmz eval resnet50_224x224 --profile onnx
```

`[cpu]` and `[gpu]` conflict (`onnxruntime` vs. `onnxruntime-gpu`); uninstall one before switching.
Dataset layout and downloads → [Datasets Guide](https://developer.deepx.ai/tech-docs/DXNN-SDK/DX-ModelZoo/02_04_Datasets_Guide/) ·
Options and output → [Model Evaluation](https://developer.deepx.ai/tech-docs/DXNN-SDK/DX-ModelZoo/02_02_Model_Evaluation/)

## Compile and Run on the NPU

Install DX-COM and the DX-RT Python package **in the same Python environment as DX-ModelZoo**. `dxmz` imports
`dx_com` and `dx_engine` directly, so a separate virtualenv (such as `~/deepx/venv-dx-compiler` from the
DX-Compiler one-line installer) is not visible to it.

```bash
pip install dx-com                             # DX-COM, x86_64 host only
pip install /usr/share/libdxrt-bin/python/dx_engine-*-cp$(python3 -c 'import sys;print(f"{sys.version_info[0]}{sys.version_info[1]}")')-*.whl

dxmz compile resnet50_224x224 --profile q-lite
dxmz eval resnet50_224x224 --profile q-lite
```

The `dx_engine` wheels ship with the `libdxrt-bin` package, which is built with `USE_ORT=ON`.
Setup details, WSL2 and GPU → [Installation](https://developer.deepx.ai/tech-docs/DXNN-SDK/DX-ModelZoo/02_01_Installation/)

<details>
<summary><b>Profiles and Environment Variables</b></summary>

| Profile | Runs on | What it does |
|---|---|---|
| `onnx` | ONNX Runtime (CPU/GPU) | Reference accuracy of the original model |
| `c-lite` | NPU (`.dxnn`) | Compile-Lite: minimal optimization (`opt_level=0`) |
| `q-lite` | NPU (`.dxnn`) | Quantize-Lite: standard EMA calibration |
| `q-pro` | NPU (`.dxnn`) | Quantize-Pro: automatic Q-PRO pipeline (`calibration.use_q_pro: true`) |

| Variable | Description |
|---|---|
| `DATA_ROOT`, `MODEL_ROOT` | Dataset and model locations, usable as `${VAR}` in YAML files |
| `DXMZ_MODEL_URL` | Base URL for automatic model download when ONNX/DXNN files are not found locally |

Every model is one YAML file with its preprocessing, postprocessing, dataset and profiles →
[YAML Configuration](https://developer.deepx.ai/tech-docs/DXNN-SDK/DX-ModelZoo/02_03_YAML_Configuration/) ·
[CLI Reference](https://developer.deepx.ai/tech-docs/DXNN-SDK/DX-ModelZoo/04_01_CLI_Reference/)

</details>

## Documentation & Support

- Guides and API reference: **[DX-ModelZoo Tech Docs](https://developer.deepx.ai/tech-docs/DXNN-SDK/DX-ModelZoo/DX-ModelZoo_Overview/)** · [Custom Models](https://developer.deepx.ai/tech-docs/DXNN-SDK/DX-ModelZoo/02_05_Custom_Models/) · [Release Notes](./RELEASE_NOTES.md)
- For technical questions, please ask in the DEEPX Developer Community: https://community.deepx.ai/c/support/5

## License

- **Source code**: released under the MIT License. See [LICENSE](./LICENSE) for terms of use.
- **Model weights** (ONNX / DXNN): each model follows the license of its original (upstream)
  project, which is **separate from the MIT license of this repository**. Many models are
  under copyleft licenses such as AGPL-3.0 (e.g. Ultralytics YOLO) or GPL-3.0 (e.g. YOLOv7),
  and some are **non-commercial only** (e.g. NVIDIA Source Code License-NC, CC-BY-NC-4.0). See [MODEL_LICENSES.md](./MODEL_LICENSES.md)
  for the original license of every model and check it before commercial use.
  The authoritative license of each model is the `license` field in its YAML under
  [`src/dx_modelzoo/models/`](./src/dx_modelzoo/models/); MODEL_LICENSES.md is generated from it.
- **Datasets** (COCO, ImageNet, ...) are not redistributed; obtain them from their original
  sources under their own terms.

Copyright © DEEPX Co., Ltd.
