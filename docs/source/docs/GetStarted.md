# Get Started

This chapter provides installation guidelines and usage examples for DeepX Open ModelZoo.

## Prerequisites

- Operating System: Ubuntu 20.04 LTS or 22.04 LTS
- Python: Version ≥ 3.11
- DeepX Components:
  - DeepX Runtime (DX-RT) ≥ 2.7.0 (compiled with USE_ORT=ON)
  - DeepX Compiler (DX-COM) ≥ 1.45.0
- (optional) GPU Requirements:
  - NVIDIA GPU (Pascal/Turing/Ampere architecture)
  - NVIDIA Driver ≥ 535.230.02
  - CUDA 12.2
  - cuDNN 9.1

Note:

- Refer to the official DX-RT and DX-COM installation guides for detailed configuration instructions.
- Using GPU acceleration is optional.
- Soft link to your actual dataset locations to `./datasets`.

## Modelzoo Installation

### Manual Installation

- Clone the DeepX Open Modelzoo Repository
  ```
  git clone https://github.com/DEEPX-AI/dx-modelzoo.git
  ```
- Install package

  ```
  cd dx-modelzoo
  ./setup.sh
  ```

- Install DeepX python package (refer to the official DX-RT and DX-COM installation guides, make sure dx_engine performs properly.)
  ```
  cd ~/dx_rt/python_package
  pip uninstall dx-engine
  pip install .
  python -c "from dx_engine import InferenceEngine"
  ```
- To enable the GPU acceleration (option)
  ```
  pip install onnxruntime-gpu==1.20.1
  ```

## Usage Guide

### CLI (command line interface)

- **Model list**

  - Description:

    Command to display a list of featured models:

  - Command:
    ```
    dxmz models
    ```

- **Performance evaluation**

  - Description:

    Command to measure the accuracy of the specific model:

  - Command Syntax：
    ```
    dxmz eval <model_name> [--onnx <path> | --dxnn <path>] --data_dir <dataset_path>
    ```
  - Runtime options:

    ```
    # To use onnx runtime:
    dxmz eval <Model Name> --onnx <onnx file path> --data_dir <dataset root dir path>

    # To use DX-Runtime:
    dxmz eval <Model Name> --dxnn <dxnn file path> --data_dir <dataset root dir path>
    ```

- **Integrated benchmarking tool**

  - Description:

    An out-of-box benchmarking tool is provided to automatically download DXNN and ONNX models from sdk.deepx.ai, and conduct evaluations of each featured model. It takes approximately 5~ hours to complete. After that, you can find the performance comperison report in `./benchmark_report.json`.

  - Command Syntax：
    ```
    dxmz benchmark --all
    ```
  - Runtime options: (No comperison report to export, when evaluate indivial runtime)

    ```
    # To use onnx runtime:
    dxmz benchmark --onnxrt

    # To use DX-Runtime:
    dxmz benchmark --dxrt
    ```
