# YOLOv8 Retraining Environment

This project provides a Docker environment for retraining the YOLOv8 model. It is based on NVIDIA's official PyTorch image to ensure full GPU acceleration support, allowing you to easily set up a reproducible and isolated development environment.

## 📝 Overview

The `Dockerfile` performs the following steps:

-   **Base Image**: Uses `nvcr.io/nvidia/pytorch` as the base, which comes with CUDA and cuDNN pre-installed.
-   **Install Dependencies**: Installs essential packages like `git` and `python3-venv`.
-   **Clone Repository**: Clones the latest Ultralytics source code directly from GitHub.
-   **Set Up Virtual Environment**: Configures a Python virtual environment (`venv`) to isolate dependencies.
-   **Install Ultralytics**: Installs the `ultralytics` package using `pip`.
-   **Create Non-Root User**: Creates a non-root user (`deepx`) for enhanced security.

## ✅ Prerequisites

Before building and running this Docker image, you must have the following software installed on your system:

-   [Docker](https://docs.docker.com/engine/install/)
-   [NVIDIA GPU Drivers](https://www.nvidia.com/Download/index.aspx)
-   [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

To verify that the NVIDIA Container Toolkit is installed correctly, run the following command. It should execute without errors and display your GPU information.

```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

---

## 🚀 How to Build

1.  Open a terminal in the same directory where the `Dockerfile` is located.
2.  Run the following command to build the Docker image:

    ```bash
    docker build -t yolov8-retraining .
    ```

    -   `-t yolov8-retraining`: Tags the built image with the name `yolov8-retraining`. You can change this to any name you prefer.

---

## ▶️ How to Run

Use the following command to start a container from the built image. You must replace the placeholder paths for volumes with the actual paths on your local machine.

```bash
docker run --gpus all -it --rm --ipc=host \
    -v /path/to/your/datasets:/app/datasets \
    -v /path/to/your/results:/app/ultralytics/runs \
    yolov8-retraining bash
```

-   `--gpus all`: Assigns all available host GPUs to the container.
-   `-it`: Allocates an interactive TTY (terminal).
-   `--rm`: Automatically removes the container when it exits.
-   `--ipc=host`: Sets the IPC (Inter-Process Communication) namespace to the host's. This can prevent shared memory issues when using PyTorch's DataLoader.
-   `-v /path/to/your/datasets:/app/datasets`: Mounts your local dataset directory into the container. **Remember to change the host path.**
-   `-v /path/to/your/results:/app/ultralytics/runs`: Mounts a local directory to save the training results (weights, logs, etc.). **Remember to change the host path.**
-   `yolov8-retraining`: The name of the image to run.
-   `bash`: Starts an interactive bash shell inside the container.

---

## ✨ Training Your Model

Training a custom YOLOv8 model is a two-step process: preparing the dataset and running the training command.

### Step 1: Prepare Your Dataset

YOLOv8 requires a specific dataset format. You need a directory of images and a corresponding directory of label files, along with a YAML configuration file that defines the dataset.

1.  **Organize Your Data Folders**: Your dataset directory should contain `images` and `labels` folders for both your training and validation sets (e.g., `train/images`, `train/labels`, `val/images`, `val/labels`).

2.  **Create Label Files**: For each image in the `images` folder, there must be a corresponding `.txt` file in the `labels` folder with the same name. Each row in the `.txt` file represents a single bounding box and must be in the following format:

    ```
    <class_index> <x_center> <y_center> <width> <height>
    ```
    -   `<class_index>`: An integer representing the class, starting from 0. This index must correspond to the order in the `names` list in your `.yaml` file.
    -   `<x_center> <y_center> <width> <height>`: Floating-point values between 0 and 1, representing the bounding box's center coordinates, width, and height, all normalized relative to the image's dimensions.

3.  **Create a Dataset YAML file**: Create a `.yaml` file (e.g., `my_dataset.yaml`) that describes your dataset's structure and class names. The paths in this file should correspond to the paths inside the container.

    **Example `my_dataset.yaml`:**
    ```yaml
    # The root path to your dataset inside the container
    path: /app/datasets/your_dataset_name
    
    # Paths to training and validation image directories (relative to 'path')
    train: images/train
    val: images/val
    
    # Class information
    nc: 2  # number of classes
    names: ['person', 'car'] # list of class names. 'person' is index 0, 'car' is index 1.
    ```

4.  **Place Your Data**: Ensure your complete dataset folder (containing the images, labels, and the `.yaml` file) is located inside the local directory you mounted to `/app/datasets`.

For a complete guide on creating custom datasets, please refer to the official **[Ultralytics Dataset Guide](https://docs.ultralytics.com/datasets/detect/)**.

### Step 2: Start Training

Once your dataset is prepared and mounted, you can start the training process from within the container's shell. Use the `yolo` command, pointing to your custom dataset's YAML file and a base model.

```bash
# Inside the container shell
yolo detect train data=/app/datasets/my_dataset.yaml model=yolov8s.pt name=my_yolov8_retrain epochs=100 batch=16
```

-   `data`: Path to your dataset's `.yaml` file inside the container.
-   `model`: The base model to start from (e.g., `yolov8s.pt`, `yolov8m.pt`).
-   `name`: The name for your training run. Results will be saved in a folder with this name.
-   `epochs`: Number of training epochs.
-   `batch`: Batch size. Adjust based on your GPU memory.

The training results will be saved in `/app/ultralytics/runs`, which is mounted to your local `/path/to/your/results` directory, ensuring your trained models and logs are preserved after the container exits.

For more detailed information on command-line arguments and advanced training features, please consult the official **[Ultralytics Train Mode Guide](https://docs.ultralytics.com/modes/train/)**.