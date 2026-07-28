# Installation

Step-by-step instructions for installing DX-ModelZoo and its dependencies on Linux systems.

## Prerequisites

Before installing DX-ModelZoo, ensure you have the following:

- **Python**: 3.8+
- **DeepX Components**:
  - [DeepX Runtime (DX-RT)](https://github.com/DEEPX-AI/dx-rt) ≥ 3.0.0 (compiled with `USE_ORT=ON`)
  - [DeepX Compiler (DX-Compiler)](https://github.com/DEEPX-AI/dx-compiler) ≥ 2.4.0

!!! note "See Also"
    - [Quick Start](quickstart.md) - Get started with DX-ModelZoo
    - [Model Evaluation](../guides/evaluation.md) - Evaluate models after installation

!!! note "Windows Users"
    DX-ModelZoo runs on Windows via WSL2 (Windows Subsystem for Linux).

    To install WSL2:
    ```bash
    # Run in PowerShell (admin mode)
    wsl --install
    ```
    Then reboot and launch Ubuntu from the Start menu. See [Microsoft's WSL documentation](https://docs.microsoft.com/en-us/windows/wsl/install) for details.

!!! note "GPU Requirements (Optional)"
    For GPU inference:

    - NVIDIA GPU (Pascal/Turing/Ampere architecture)
    - NVIDIA Driver ≥ 535.230.02
    - CUDA 12.2
    - cuDNN 9.1

## Installation Overview

**Required Steps:**

1. [Install DeepX Components](#step-1-install-deepx-components) - DX-RT & DX-Compiler
2. [Clone Repository](#step-2-clone-repository) - Get the source code
3. [Basic Installation](#step-3-basic-installation-cpu-inference) - CPU inference setup

**Optional Steps (choose as needed):**

- [GPU Inference](#optional-gpu-inference-cuda) - For CUDA acceleration
- [Development Environment](#optional-development-environment) - For contributors
- [Documentation Build](#optional-documentation-build) - To build docs locally
- [Full Installation](#optional-full-installation-cpu-dev-docs) - All features at once

---

## Step 1: Install DeepX Components

First, install the required DeepX components:

```bash
# Install DX-RT python package
pip install dx_engine-*.whl

# Install DX-Compiler python package
pip install dx_com-*.whl
```

!!! note "NOTE"
    Refer to the official DX-RT and DX-Compiler installation guides for detailed setup instructions.

## Step 2: Clone Repository

```bash
git clone https://github.com/DEEPX-AI/dx-modelzoo.git
cd dx-modelzoo
```

## Step 3: Basic Installation (CPU Inference)

```bash
pip install -e ".[cpu]"
```

This installs ONNX Runtime CPU and all core dependencies.

## Optional: GPU Inference (CUDA)

```bash
pip install -e ".[gpu]"
```

!!! warning "CUDA Environment Setup"
    GPU inference requires:

    - **NVIDIA Driver** ≥ 535.230.02
    - **CUDA** 12.2
    - **cuDNN** 9.1
    - **GPU Architecture**: Pascal/Turing/Ampere

    After installing `.[gpu]`, add NVIDIA library paths to `LD_LIBRARY_PATH`:

    ```bash
    # Add to ~/.zshrc or ~/.bashrc
    export LD_LIBRARY_PATH=$(python -c "import nvidia.cublas.lib; print(nvidia.cublas.lib.__path__[0])"):$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=$(python -c "import nvidia.cudnn.lib; print(nvidia.cudnn.lib.__path__[0])"):$LD_LIBRARY_PATH
    ```

## Optional: Development Environment

```bash
pip install -e ".[dev]"   # Includes pytest, ruff
```

## Optional: Documentation Build

```bash
pip install -e ".[docs]"  # Includes mkdocs, mkdocs-material
mkdocs serve              # Preview at http://localhost:8000
```

## Optional: Full Installation (cpu + dev + docs)

```bash
pip install -e ".[all]"
```

## Optional Extras Summary

| Extra | Packages | Purpose |
|-------|----------|---------|
| `cpu` | `onnxruntime>=1.15` | CPU-based ONNX inference |
| `gpu` | `onnxruntime-gpu>=1.17` | GPU-based ONNX inference |
| `dev` | `pytest>=7.0`, `ruff>=0.1` | Testing & linting |
| `docs` | `mkdocs`, `mkdocs-material`, `mkdocstrings` | Documentation build |
| `all` | `cpu` + `dev` + `docs` | Everything |

## Core Dependencies

DX-ModelZoo keeps its core dependencies minimal:

```
pyyaml>=5.0       # YAML parsing
numpy>=1.20       # Numerical computation
Pillow>=8.0       # Image processing
opencv-python>=4.5 # Image preprocessing
tqdm>=4.60        # Progress bars
loguru>=0.6       # Logging
typer>=0.9        # CLI
rich>=12.0        # Terminal output
```

!!! note "PyTorch stack is part of the core runtime"
    `torch` is a core dependency in `pyproject.toml`, and the source tree also
    relies on `torchvision` APIs for preprocessing/data loading paths. Do not
    treat PyTorch as an optional compile-only add-on.

## Alternative Package Manager

### Using uv

For faster dependency resolution, you can use `uv` instead of `pip`:

```bash
uv pip install -e ".[cpu]"
```

!!! note "Dependency resolution is much faster with `uv`"
    ```bash
    pip install uv               # Install uv
    uv pip install -e ".[gpu]"  # Fast installation
    ```
