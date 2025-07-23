## Introduction

DeepX Open Modelzoo provides developers with an effortless experience in utilizing DeepX NPUs for widely applied tasks, covering image classification, object detection, semantic segmentation, face id and image denoising.

All featured neural network models are provided with pre-trained ONNX models, configuration json files, and pre-compiled INT8 binaries named DXNN(DeepX Neural Network). Comprehensive benchmark tools are available for comparing the performance of models on DeepX NPUs with that on GPUs or CPUs.

Additionally, developers can also compile featured ONNXs to DXNNs via DX_COM, which enables rapid application development accelerated by DeepX NPUs.

As of 2025-02-14, developers can discover following featured models with DeepX NPUs. The number of models covered by the DeepX Model Zoo will continuously expand as the DeepX SDK is upated.

<!-- | **Task Type** | **Model List**  |
|:--------------|:----------------|
| **[Image Classification](pplx://action/followup)** | DenseNet121, DenseNet161, DnCNN_15, DnCNN_25, DnCNN_50, EfficientNetB2, EfficientNetV2S, HarDNet39DS, MobileNetV1, MobileNetV2, MobileNetV3Large,  MobileNetV3Small, AlexNet |
| **[Object Detection](pplx://action/followup)**      | YoloV3, YoloV5L, YoloV5M, YoloV5N, YoloV5S, YoloV7, YoloV7E6, YoloV7Tiny, YoloV8L, YoloV9C, YoloV9S, YoloV9T, YoloXS  |
| **[Semantic Segmentation](pplx://action/followup)** | BiSeNetV1, BiSeNetV2, DeepLabV3PlusDRN, DeepLabV3PlusMobileNetV2, DeepLabV3PlusMobilenet, DeepLabV3PlusResNet101, DeepLabV3PlusResNet50, DeepLabV3PlusResnet |
| **[Face ID](pplx://action/followup)**              | YOLOv5m_Face, YOLOv5s_Face, YOLOv7_Face, YOLOv7_TTA_Face, YOLOv7_W6_Face, YOLOv7_W6_TTA_Face, YOLOv7s_Face |
| **[Image De-Noising](pplx://action/followup)**     | DnCNN_15, DnCNN_25, DnCNN_50 | -->

**[Image Classification](pplx://action/followup)**:
DenseNet121, DenseNet161, DnCNN_15, DnCNN_25, DnCNN_50, EfficientNetB2, EfficientNetV2S, HarDNet39DS, MobileNetV1, MobileNetV2, MobileNetV3Large, MobileNetV3Small, AlexNet

**[Object Detection](pplx://action/followup)**:
YoloV3, YoloV5L, YoloV5M, YoloV5N, YoloV5S, YoloV7, YoloV7E6, YoloV7Tiny, YoloV8L, YoloV9C, YoloV9S, YoloV9T, YoloXS

**[Semantic Segmentation](pplx://action/followup)**:
BiSeNetV1, BiSeNetV2, DeepLabV3PlusDRN, DeepLabV3PlusMobileNetV2, DeepLabV3PlusMobilenet, DeepLabV3PlusResNet101, DeepLabV3PlusResNet50, DeepLabV3PlusResnet

**[Face ID](pplx://action/followup)**:
YOLOv5m_Face, YOLOv5s_Face, YOLOv7_Face, YOLOv7_TTA_Face, YOLOv7_W6_Face, YOLOv7_W6_TTA_Face, YOLOv7s_Face

**[Image De-Noising](pplx://action/followup)**:
DnCNN_15, DnCNN_25, DnCNN_50
