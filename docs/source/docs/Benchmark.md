# Benchmark

We offer benchmarking tools to evaluate the featured models across platforms including DeepX NPU, GPU and CPU. For DeepX NPU, you can determine the FPS of models using CLIs and evaluate their accuracy with supported datasets.

## Regression mode

Command：

```
./bin/run_model -m <dxnn file>
```

Expected output:

```
Start benchmark.
Completed benchmark.
Data test: Sequential
-----------------------------------
model
  Inference time : 4.41787ms
  FPS : 226.354
  Data Integrity Check : SKIP (0 pass, 0 fail )
-----------------------------------
```

## Evaluation mode

Quantitative performance evaluation across runtimes:

- With ONNX Runtime:
  ```
  dxmz eval <Model Name> --onnx <onnx file path> --data_dir <dataset root dir path>
  ```
- With DeepX Runtime(DX-RT):
  ```
  dxmz eval <Model Name> --dxnn <dxnn file path> --data_dir <dataset root dir path>
  ```
<!-- 
### Integrated benchmarking

An out-of-box benchmarking tool is provided to automatically download DXNN and ONNX models from sdk.deepx.ai, and conduct evaluations of each featured model. It takes approximately 5~ hours to complete. After that, you can find the performance comperison report in `./benchmark_report.json`.

- Command Syntax：

  ```
  dxmz benchmark --all
  ```

- `Deepx_benchmark.json` containing:

  - Meta information
  - Model performance metrics
  - Runtime FPS

- Runtime options: (No comperison report to export, when evaluate indivial runtime)

  ```
  # To use onnx runtime:
  dxmz benchmark --onnxrt

  # To use DX-Runtime:
  dxmz benchmark --dxrt
  ``` -->
