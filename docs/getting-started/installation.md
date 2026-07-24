# Installation

## Basic Installation (CPU Inference)

```bash
pip install -e ".[cpu]"
```

This installs ONNX Runtime CPU and all core dependencies.

## GPU Inference (CUDA)

```bash
pip install -e ".[gpu]"
```

!!! warning "CUDA Environment Setup"
    GPU inference requires nvidia library paths in `LD_LIBRARY_PATH`:

    ```bash
    # Add to ~/.zshrc or ~/.bashrc
    export LD_LIBRARY_PATH=$(python -c "import nvidia.cublas.lib; print(nvidia.cublas.lib.__path__[0])"):$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=$(python -c "import nvidia.cudnn.lib; print(nvidia.cudnn.lib.__path__[0])"):$LD_LIBRARY_PATH
    ```

## Development Environment

```bash
pip install -e ".[dev]"   # Includes pytest, ruff
```

## Documentation Build

```bash
pip install -e ".[docs]"  # Includes mkdocs, mkdocs-material
mkdocs serve              # Preview at http://localhost:8000
```

## Full Installation (cpu + dev + docs)

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

DX ModelZoo keeps its core dependencies minimal:

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

## System Requirements

- **Python**: 3.8+
- **OS**: Linux (recommended), macOS, Windows
- **Package manager**: `pip` or `uv` (recommended)

### Using uv

```bash
uv pip install -e ".[cpu]"
```

!!! tip "Dependency resolution is much faster with `uv`"
    ```bash
    pip install uv        # Install uv
    uv pip install -e ".[gpu]"  # Fast installation
    ```
