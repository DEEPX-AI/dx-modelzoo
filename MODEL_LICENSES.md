<!-- Generated from the `license` field of each model YAML. Do not edit by hand. -->

# Model Licenses

The DX-ModelZoo **source code** is licensed under the MIT License (see [LICENSE](LICENSE)).
**Model weights** (ONNX / DXNN) are derived from upstream projects and remain subject to the
**original license of each model**, listed below. This license is separate from the MIT license
of this repository. Check the original license before commercial use.

This table is generated from the `license` field of each model YAML under
[`src/dx_modelzoo/models/`](src/dx_modelzoo/models/), which is the authoritative source.
Each YAML lists the license of the upstream source code (`license.model`) and of the
pretrained weights (`license.weights`); the weights license decides how a model may be used.

- **Non-commercial**: must not be used in commercial products.
- **Copyleft** (AGPL / GPL / LGPL): whoever distributes the model takes on the obligations of
  the original license (e.g. providing source). Under AGPL, serving the model over a network
  also counts.

Datasets used for evaluation (COCO, ImageNet, etc.) are **not redistributed** by this repository;
obtain them from their original sources under their own terms.

## Non-commercial models

> **Warning**: the following models are licensed for non-commercial / research use only.

| Model | Task | Weights license | Code license | Upstream |
|---|---|---|---|---|
| `depthanythingv2-vitb_224x224` | depth_estimation | CC-BY-NC-4.0 | Apache-2.0 | [link](https://github.com/DepthAnything/Depth-Anything-V2) |
| `depthanythingv2-vitl_224x224` | depth_estimation | CC-BY-NC-4.0 | Apache-2.0 | [link](https://github.com/DepthAnything/Depth-Anything-V2) |
| `retinaface_mobilenetv1_736x1280` | face_detection | InsightFace Model License (Non-Commercial Research) | MIT | [link](https://github.com/deepinsight/insightface) |
| `scrfd-10g_640x640` | face_detection | InsightFace Model License (Non-Commercial Research) | MIT | [link](https://github.com/deepinsight/insightface) |
| `scrfd-2.5g_640x640` | face_detection | InsightFace Model License (Non-Commercial Research) | MIT | [link](https://github.com/deepinsight/insightface) |
| `scrfd-500m_640x640` | face_detection | InsightFace Model License (Non-Commercial Research) | MIT | [link](https://github.com/deepinsight/insightface) |
| `arcface_iresnet100_112x112_ms1m` | face_recognition | InsightFace Model License (Non-Commercial Research) | MIT | [link](https://github.com/deepinsight/insightface) |
| `arcface_iresnet50_112x112_ms1m` | face_recognition | InsightFace Model License (Non-Commercial Research) | MIT | [link](https://github.com/deepinsight/insightface) |
| `arcface_mobilefacenet_112x112` | face_recognition | InsightFace Model License (Non-Commercial Research) | MIT | [link](https://github.com/deepinsight/insightface) |
| `arcface_resnet50_112x112` | face_recognition | InsightFace Model License (Non-Commercial Research) | MIT | [link](https://github.com/deepinsight/insightface) |
| `regnet-y16gf_384x384` | image_classification | CC-BY-NC-4.0 | BSD-3-Clause | [link](https://github.com/pytorch/vision) |
| `regnet-y32gf_384x384` | image_classification | CC-BY-NC-4.0 | BSD-3-Clause | [link](https://github.com/pytorch/vision) |
| `vit-l-p16_512x512_swag` | image_classification | CC-BY-NC-4.0 | BSD-3-Clause | [link](https://pytorch.org/vision/stable/models/generated/torchvision.models.vit_l_16.html) |
| `superpoint_480x640` | keypoint_detection | Magic Leap Custom License (Academic/Noncommercial) | Magic Leap Custom License (Academic/Noncommercial) | [link](https://github.com/magicleap/SuperPointPretrainedNetwork) |
| `zerodce-pp_400x600` | low_light_enhancement | CC-BY-NC-4.0 | CC-BY-NC-4.0 | [link](https://github.com/Li-Chongyi/Zero-DCE_extension) |
| `zerodce_400x600` | low_light_enhancement | CC-BY-NC-4.0 | CC-BY-NC-4.0 | [link](https://github.com/Li-Chongyi/Zero-DCE) |
| `dope-hope-ketchup_480x640` | object_pose_estimation | NVIDIA Source Code License-NC | NVIDIA Source Code License-NC | [link](https://github.com/NVlabs/Deep_Object_Pose) |
| `deepmar_resnet18_224x224` | person_attribute | Custom License (Academic Research Only) | Custom License (Academic Research Only) | [link](https://github.com/dangweili/pedestrian-attribute-recognition-pytorch) |
| `deepmar_resnet50_224x224` | person_attribute | Custom License (Academic Research Only) | Custom License (Academic Research Only) | [link](https://github.com/dangweili/pedestrian-attribute-recognition-pytorch) |
| `segformer_mit-b0_512x1024` | semantic_segmentation | NVIDIA Source Code License-NC | NVIDIA Source Code License-NC | [link](https://github.com/NVlabs/SegFormer) |
| `clip-img_vit-b16-quickgelu_224x224_metaclip-fullcc` | zero_shot_image_classification | CC-BY-NC-4.0 | MIT | [link](https://github.com/mlfoundations/open_clip) |
| `clip-img_vit-l14-quickgelu_224x224_dfn2b` | zero_shot_image_classification | Apple Machine Learning Research Model License | MIT | [link](https://github.com/mlfoundations/open_clip) |

## All models

| Model | Task | Weights license | Code license | Note | Upstream | Evaluation dataset |
|---|---|---|---|---|---|---|
| `sfa3d_608x608` | 3d_object_detection | MIT | MIT |  | [link](https://github.com/maudzung/SFA3D) | KITTI |
| `efficientad-m-autoencoder_256x256` | anomaly_detection | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/nelson1425/EfficientAD) | MVTecAD |
| `efficientad-m-student_256x256` | anomaly_detection | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/nelson1425/EfficientAD) | MVTecAD |
| `efficientad-m-teacher_256x256` | anomaly_detection | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/nelson1425/EfficientAD) | MVTecAD |
| `patchcore_224x224` | anomaly_detection | BSD-3-Clause | Apache-2.0 |  | [link](https://github.com/amazon-science/patchcore-inspection) | MVTecAD |
| `depthanythingv2-vitb_224x224` | depth_estimation | CC-BY-NC-4.0 | Apache-2.0 | **Non-commercial use only** | [link](https://github.com/DepthAnything/Depth-Anything-V2) | NYUDepthv2 |
| `depthanythingv2-vitl_224x224` | depth_estimation | CC-BY-NC-4.0 | Apache-2.0 | **Non-commercial use only** | [link](https://github.com/DepthAnything/Depth-Anything-V2) | NYUDepthv2 |
| `depthanythingv2-vits_224x224` | depth_estimation | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/DepthAnything/Depth-Anything-V2) | NYUDepthv2 |
| `fastdepth_224x224` | depth_estimation | MIT | MIT |  | [link](https://github.com/dwofk/fast-depth) | NYUDepthv2 |
| `scdepthv3_256x320` | depth_estimation | GPL-3.0 | GPL-3.0 | Copyleft | [link](https://github.com/JiawangBian/sc_depth_pl) | NYUDepthv2 |
| `yolo26-depth-l_768x768` | depth_estimation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | NYUDepthv2 |
| `yolo26-depth-m_768x768` | depth_estimation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | NYUDepthv2 |
| `yolo26-depth-n_768x768` | depth_estimation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | NYUDepthv2 |
| `yolo26-depth-s_768x768` | depth_estimation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | NYUDepthv2 |
| `yolo26-depth-x_768x768` | depth_estimation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | NYUDepthv2 |
| `faceattr_resnetv1-18_218x178` | face_attribute | MIT | MIT |  | [link](https://github.com/d-li14/face-attribute-prediction) | CelebA |
| `retinaface_mobilenet-0.25_640x640` | face_detection | MIT | MIT |  | [link](https://github.com/biubug6/Pytorch_Retinaface) | WiderFace |
| `retinaface_mobilenetv1_736x1280` | face_detection | InsightFace Model License (Non-Commercial Research) | MIT | **Non-commercial use only** | [link](https://github.com/deepinsight/insightface) | WiderFace |
| `scrfd-10g_640x640` | face_detection | InsightFace Model License (Non-Commercial Research) | MIT | **Non-commercial use only** | [link](https://github.com/deepinsight/insightface) | WiderFace |
| `scrfd-2.5g_640x640` | face_detection | InsightFace Model License (Non-Commercial Research) | MIT | **Non-commercial use only** | [link](https://github.com/deepinsight/insightface) | WiderFace |
| `scrfd-500m_640x640` | face_detection | InsightFace Model License (Non-Commercial Research) | MIT | **Non-commercial use only** | [link](https://github.com/deepinsight/insightface) | WiderFace |
| `ulfgfd-rfb_240x320` | face_detection | MIT | MIT |  | [link](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) | WiderFace |
| `ulfgfd-rfb_240x320_nopp` | face_detection | MIT | MIT |  | [link](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) | WiderFace |
| `ulfgfd-rfb_480x640` | face_detection | MIT | MIT |  | [link](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) | WiderFace |
| `ulfgfd-slim_240x320` | face_detection | MIT | MIT |  | [link](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) | WiderFace |
| `ulfgfd-slim_240x320_nopp` | face_detection | MIT | MIT |  | [link](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) | WiderFace |
| `yolov5-m-face_640x640` | face_detection | GPL-3.0 | GPL-3.0 | Copyleft | [link](https://github.com/deepcam-cn/yolov5-face) | WiderFace |
| `yolov5-n-face_640x640` | face_detection | GPL-3.0 | GPL-3.0 | Copyleft | [link](https://github.com/deepcam-cn/yolov5-face) | WiderFace |
| `yolov5-s-face_640x640` | face_detection | GPL-3.0 | GPL-3.0 | Copyleft | [link](https://github.com/deepcam-cn/yolov5-face) | WiderFace |
| `yolov7-face_640x640` | face_detection | GPL-3.0 | GPL-3.0 | Copyleft | [link](https://github.com/derronqi/yolov7-face) | WiderFace |
| `yolov7-lite-t-face_640x640` | face_detection | GPL-3.0 | GPL-3.0 | Copyleft | [link](https://github.com/derronqi/yolov7-face) | WiderFace |
| `yolov7-s-face_640x640` | face_detection | GPL-3.0 | GPL-3.0 | Copyleft | [link](https://github.com/derronqi/yolov7-face) | WiderFace |
| `yolov7-w6-face_1280x1280_tta` | face_detection | GPL-3.0 | GPL-3.0 | Copyleft | [link](https://github.com/derronqi/yolov7-face) | WiderFace |
| `yolov7-w6-face_960x960` | face_detection | GPL-3.0 | GPL-3.0 | Copyleft | [link](https://github.com/derronqi/yolov7-face) | WiderFace |
| `3ddfa-v2_mobilenet-0.5_120x120` | face_landmark | MIT | MIT |  | [link](https://github.com/cleardusk/3DDFA_V2) | AFLW20003D |
| `3ddfa-v2_mobilenetv1_120x120` | face_landmark | MIT | MIT |  | [link](https://github.com/cleardusk/3DDFA_V2) | AFLW20003D |
| `arcface_iresnet100_112x112_ms1m` | face_recognition | InsightFace Model License (Non-Commercial Research) | MIT | **Non-commercial use only** | [link](https://github.com/deepinsight/insightface) | LFW |
| `arcface_iresnet50_112x112_ms1m` | face_recognition | InsightFace Model License (Non-Commercial Research) | MIT | **Non-commercial use only** | [link](https://github.com/deepinsight/insightface) | LFW |
| `arcface_mobilefacenet_112x112` | face_recognition | InsightFace Model License (Non-Commercial Research) | MIT | **Non-commercial use only** | [link](https://github.com/deepinsight/insightface) | LFW |
| `arcface_resnet50_112x112` | face_recognition | InsightFace Model License (Non-Commercial Research) | MIT | **Non-commercial use only** | [link](https://github.com/deepinsight/insightface) | LFW |
| `mediapipe-hand-detector_192x192` | hand_detection | Apache-2.0 | Apache-2.0 |  | [link](https://google.github.io/mediapipe/solutions/hands) | HandKeypointsDetection |
| `mediapipe-hands-lite_224x224` | hand_landmark | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/google-ai-edge/mediapipe) | HandKeypoints |
| `alexnet_224x224` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `beit-l-p16_224x224` | image_classification | MIT | MIT |  | [link](https://github.com/microsoft/unilm/tree/master/beit) | ILSVRC2012 |
| `beit-l-p16_384x384` | image_classification | MIT | MIT |  | [link](https://github.com/microsoft/unilm/tree/master/beit) | ILSVRC2012 |
| `casvit-m_224x224` | image_classification | MIT | MIT |  | [link](https://github.com/Tianfang-Zhang/CAS-ViT) | ILSVRC2012 |
| `casvit-t_224x224` | image_classification | MIT | MIT |  | [link](https://github.com/Tianfang-Zhang/CAS-ViT) | ILSVRC2012 |
| `deit-b_224x224` | image_classification | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/facebookresearch/deit) | ILSVRC2012 |
| `deit-b_224x224_distilled` | image_classification | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/facebookresearch/deit) | ILSVRC2012 |
| `deit-b_384x384` | image_classification | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/facebookresearch/deit) | ILSVRC2012 |
| `deit-b_384x384_distilled` | image_classification | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/facebookresearch/deit) | ILSVRC2012 |
| `deit-s_224x224` | image_classification | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/facebookresearch/deit) | ILSVRC2012 |
| `deit-s_224x224_distilled` | image_classification | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/facebookresearch/deit) | ILSVRC2012 |
| `deit-t_224x224` | image_classification | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/facebookresearch/deit) | ILSVRC2012 |
| `deit-t_224x224_distilled` | image_classification | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/facebookresearch/deit) | ILSVRC2012 |
| `densenet121_224x224` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `densenet161_224x224` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `densenet169_224x224` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `densenet201_224x224` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `efficientformer-l3_224x224` | image_classification | MIT | MIT |  | [link](https://github.com/snap-research/EfficientFormer) | ILSVRC2012 |
| `efficientformer-l7_224x224` | image_classification | MIT | MIT |  | [link](https://github.com/snap-research/EfficientFormer) | ILSVRC2012 |
| `efficientformerv2-l_224x224` | image_classification | MIT | MIT |  | [link](https://github.com/snap-research/EfficientFormer) | ILSVRC2012 |
| `efficientnet-b2_288x288` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `efficientnet-b3_300x300` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `efficientnet-b4_380x380` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `efficientnet-b5_456x456` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `efficientnet-b6_528x528` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `efficientnet-b7_600x600` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `efficientnet-lite0_224x224` | image_classification | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/huggingface/pytorch-image-models) | ILSVRC2012 |
| `efficientnet-lite1_240x240` | image_classification | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/huggingface/pytorch-image-models) | ILSVRC2012 |
| `efficientnet-lite2_260x260` | image_classification | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/huggingface/pytorch-image-models) | ILSVRC2012 |
| `efficientnet-lite3_300x300` | image_classification | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/huggingface/pytorch-image-models) | ILSVRC2012 |
| `efficientnet-lite4_380x380` | image_classification | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/huggingface/pytorch-image-models) | ILSVRC2012 |
| `efficientnetv2-l_480x480` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `efficientnetv2-m_480x480` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `efficientnetv2-s_384x384` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `fastvit-ma36_256x256` | image_classification | Apple Sample Code License | Apple Sample Code License |  | [link](https://github.com/apple-aiml-research/ml-fastvit) | ILSVRC2012 |
| `fastvit-sa24_256x256` | image_classification | Apple Sample Code License | Apple Sample Code License |  | [link](https://github.com/apple-aiml-research/ml-fastvit) | ILSVRC2012 |
| `fastvit-sa36_256x256` | image_classification | Apple Sample Code License | Apple Sample Code License |  | [link](https://github.com/apple-aiml-research/ml-fastvit) | ILSVRC2012 |
| `ghostnet_224x224` | image_classification | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/ghostnet_pytorch) | ILSVRC2012 |
| `hardnet39ds_224x224` | image_classification | MIT | MIT |  | [link](https://github.com/PingoLH/Pytorch-HarDNet) | ILSVRC2012 |
| `hardnet68_224x224` | image_classification | MIT | MIT |  | [link](https://github.com/PingoLH/Pytorch-HarDNet) | ILSVRC2012 |
| `inceptionv1_224x224` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `levit-128_224x224` | image_classification | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/facebookresearch/LeViT) | ILSVRC2012 |
| `levit-128s_224x224` | image_classification | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/facebookresearch/LeViT) | ILSVRC2012 |
| `levit-192_224x224` | image_classification | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/facebookresearch/LeViT) | ILSVRC2012 |
| `levit-256_224x224` | image_classification | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/facebookresearch/LeViT) | ILSVRC2012 |
| `levit-384_224x224` | image_classification | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/facebookresearch/LeViT) | ILSVRC2012 |
| `mnasnet-0.5_224x224` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `mnasnet-0.75_224x224` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `mnasnet-1.0_224x224` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `mnasnet-1.3_224x224` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `mobilenetv1_224x224` | image_classification | MIT | MIT |  | [link](https://github.com/osmr/imgclsmob) | ILSVRC2012 |
| `mobilenetv2_224x224` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `mobilenetv3-large_224x224` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `pphgnet-base_224x224` | image_classification | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/PaddlePaddle/PaddleClas/blob/develop/ppcls/arch/backbone/legendary_models/pp_hgnet.py) | ILSVRC2012 |
| `pphgnet-small_224x224` | image_classification | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/PaddlePaddle/PaddleClas/blob/develop/ppcls/arch/backbone/legendary_models/pp_hgnet.py) | ILSVRC2012 |
| `pphgnet-tiny_224x224` | image_classification | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/PaddlePaddle/PaddleClas/blob/develop/ppcls/arch/backbone/legendary_models/pp_hgnet.py) | ILSVRC2012 |
| `pphgnetv2-b0_224x224` | image_classification | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/PaddlePaddle/PaddleClas/blob/develop/ppcls/arch/backbone/legendary_models/pp_hgnet_v2.py) | ILSVRC2012 |
| `pphgnetv2-b1_224x224` | image_classification | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/PaddlePaddle/PaddleClas/blob/develop/ppcls/arch/backbone/legendary_models/pp_hgnet_v2.py) | ILSVRC2012 |
| `pphgnetv2-b2_224x224` | image_classification | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/PaddlePaddle/PaddleClas/blob/develop/ppcls/arch/backbone/legendary_models/pp_hgnet_v2.py) | ILSVRC2012 |
| `pphgnetv2-b3_224x224` | image_classification | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/PaddlePaddle/PaddleClas/blob/develop/ppcls/arch/backbone/legendary_models/pp_hgnet_v2.py) | ILSVRC2012 |
| `pphgnetv2-b4_224x224` | image_classification | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/PaddlePaddle/PaddleClas/blob/develop/ppcls/arch/backbone/legendary_models/pp_hgnet_v2.py) | ILSVRC2012 |
| `pphgnetv2-b5_224x224` | image_classification | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/PaddlePaddle/PaddleClas/blob/develop/ppcls/arch/backbone/legendary_models/pp_hgnet_v2.py) | ILSVRC2012 |
| `pphgnetv2-b6_224x224` | image_classification | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/PaddlePaddle/PaddleClas/blob/develop/ppcls/arch/backbone/legendary_models/pp_hgnet_v2.py) | ILSVRC2012 |
| `pplcnetv2-base_224x224` | image_classification | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/PaddlePaddle/PaddleClas/blob/develop/ppcls/arch/backbone/legendary_models/pp_lcnet_v2.py) | ILSVRC2012 |
| `pplcnetv2-large_224x224` | image_classification | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/PaddlePaddle/PaddleClas/blob/develop/ppcls/arch/backbone/legendary_models/pp_lcnet_v2.py) | ILSVRC2012 |
| `pplcnetv2-small_224x224` | image_classification | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/PaddlePaddle/PaddleClas/blob/develop/ppcls/arch/backbone/legendary_models/pp_lcnet_v2.py) | ILSVRC2012 |
| `regnet-x1.6gf_224x224_v1` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `regnet-x1.6gf_224x224_v2` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `regnet-x16gf_224x224` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `regnet-x3.2gf_224x224` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `regnet-x32gf_224x224` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `regnet-x400mf_224x224` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `regnet-x800mf_224x224` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `regnet-x8gf_224x224` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `regnet-y1.6gf_224x224` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `regnet-y16gf_384x384` | image_classification | CC-BY-NC-4.0 | BSD-3-Clause | **Non-commercial use only** | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `regnet-y200mf_224x224` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `regnet-y3.2gf_224x224` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `regnet-y32gf_384x384` | image_classification | CC-BY-NC-4.0 | BSD-3-Clause | **Non-commercial use only** | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `regnet-y400mf_224x224` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `regnet-y800mf_224x224` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `regnet-y8gf_224x224` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `repghost-2.0x_224x224` | image_classification | MIT | MIT |  | [link](https://github.com/ChengpengChen/RepGhost) | ILSVRC2012 |
| `repvgg-a0_320x320` | image_classification | MIT | MIT |  | [link](https://github.com/DingXiaoH/RepVGG) | ILSVRC2012 |
| `repvgg-a1_320x320` | image_classification | MIT | MIT |  | [link](https://github.com/DingXiaoH/RepVGG) | ILSVRC2012 |
| `repvgg-a2_320x320` | image_classification | MIT | MIT |  | [link](https://github.com/DingXiaoH/RepVGG) | ILSVRC2012 |
| `repvgg-b0_320x320` | image_classification | MIT | MIT |  | [link](https://github.com/DingXiaoH/RepVGG) | ILSVRC2012 |
| `repvgg-b1_320x320` | image_classification | MIT | MIT |  | [link](https://github.com/DingXiaoH/RepVGG) | ILSVRC2012 |
| `resmlp24_224x224` | image_classification | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/facebookresearch/deit/blob/main/README_resmlp.md) | ILSVRC2012 |
| `resnet101_224x224` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `resnet152_224x224` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `resnet18_224x224` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `resnet18_224x224_brecq` | image_classification | MIT | MIT |  | [link](https://github.com/yhhhli/BRECQ) | ILSVRC2012 |
| `resnet34_224x224` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `resnet50_224x224` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `resnext101-32x8d_224x224` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `resnext101-64x4d_224x224` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `resnext26-32x4d_224x224` | image_classification | MIT | MIT |  | [link](https://github.com/osmr/imgclsmob) | ILSVRC2012 |
| `resnext50-32x4d_224x224` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `resnext50-32x4d_224x224_imgclsmob` | image_classification | MIT | MIT |  | [link](https://github.com/osmr/imgclsmob) | ILSVRC2012 |
| `shufflenetv1-x1.0_224x224` | image_classification | MIT | MIT |  | [link](https://github.com/osmr/imgclsmob) | ILSVRC2012 |
| `shufflenetv2-x0.5_224x224` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `shufflenetv2-x1.0_224x224` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `shufflenetv2-x1.5_224x224` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `shufflenetv2-x2.0_224x224` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `squeezenet-1.0_224x224` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `squeezenet-1.1_224x224` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `swin-b_224x224` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://pytorch.org/vision/stable/models/generated/torchvision.models.swin_b.html#torchvision.models.swin_b) | ILSVRC2012 |
| `vgg11-bn_224x224` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `vgg11_224x224` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `vgg13-bn_224x224` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `vgg13_224x224` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `vgg16-bn_224x224` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `vgg16_224x224` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `vgg19-bn_224x224` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `vgg19_224x224` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `vit-b-p16_224x224` | image_classification | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/google-research/vision_transformer) | ILSVRC2012 |
| `vit-b-p16_224x224_bn` | image_classification | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/google-research/vision_transformer) | ILSVRC2012 |
| `vit-b-p16_384x384` | image_classification | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/google-research/vision_transformer) | ILSVRC2012 |
| `vit-b-p32_224x224` | image_classification | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/google-research/vision_transformer) | ILSVRC2012 |
| `vit-l-p16_512x512_swag` | image_classification | CC-BY-NC-4.0 | BSD-3-Clause | **Non-commercial use only** | [link](https://pytorch.org/vision/stable/models/generated/torchvision.models.vit_l_16.html) | ILSVRC2012 |
| `vit-l-p32_224x224` | image_classification | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/google-research/vision_transformer) | ILSVRC2012 |
| `vit-s-p16_224x224` | image_classification | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/google-research/vision_transformer) | ILSVRC2012 |
| `vit-t-p16_224x224` | image_classification | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/google-research/vision_transformer) | ILSVRC2012 |
| `wide-resnet101-2_224x224` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `wide-resnet50-2_224x224` | image_classification | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | ILSVRC2012 |
| `yolo26-cls-l_224x224` | image_classification | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | ILSVRC2012 |
| `yolo26-cls-m_224x224` | image_classification | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | ILSVRC2012 |
| `yolo26-cls-n_224x224` | image_classification | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | ILSVRC2012 |
| `yolo26-cls-s_224x224` | image_classification | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | ILSVRC2012 |
| `yolo26-cls-x_224x224` | image_classification | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | ILSVRC2012 |
| `yolov12-cls-l_224x224` | image_classification | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/sunsmarterjie/yolov12) | ILSVRC2012 |
| `yolov12-cls-m_224x224` | image_classification | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/sunsmarterjie/yolov12) | ILSVRC2012 |
| `yolov12-cls-n_224x224` | image_classification | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/sunsmarterjie/yolov12) | ILSVRC2012 |
| `yolov12-cls-s_224x224` | image_classification | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/sunsmarterjie/yolov12) | ILSVRC2012 |
| `yolov12-cls-x_224x224` | image_classification | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/sunsmarterjie/yolov12) | ILSVRC2012 |
| `dncnn-15_512x512` | image_denoising | MIT | MIT |  | [link](https://github.com/cszn/KAIR) | BSD68 |
| `dncnn-25_512x512` | image_denoising | MIT | MIT |  | [link](https://github.com/cszn/KAIR) | BSD68 |
| `dncnn-50_512x512` | image_denoising | MIT | MIT |  | [link](https://github.com/cszn/KAIR) | BSD68 |
| `dncnn-color_512x512` | image_denoising | MIT | MIT |  | [link](https://github.com/cszn/KAIR) | CBSD68 |
| `dncnn-gray_512x512` | image_denoising | MIT | MIT |  | [link](https://github.com/cszn/KAIR) | BSD68 |
| `ppmatting-hrnet-w48-composition_512x512` | image_matting | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.8/Matting/configs/ppmatting/ppmatting-hrnet_w48-composition.yml) | Distinctions646 |
| `ppmatting-hrnet-w48-distinctions_512x512` | image_matting | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.8/Matting/configs/ppmatting/ppmatting-hrnet_w48-distinctions.yml) | Distinctions646 |
| `clip-text_resnet50_77x512_openai` | image_retrieval | MIT | MIT |  | [link](https://github.com/mlfoundations/open_clip) | ImageNetText |
| `mask-rtdetr-hgnetv2-h-6x_640x640` | instance_segmentation | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/configs/mask_rtdetr/mask_rtdetr_hgnetv2_h_6x_coco.yml) | COCO |
| `mask-rtdetr-hgnetv2-l-6x_640x640` | instance_segmentation | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/configs/mask_rtdetr/mask_rtdetr_hgnetv2_l_6x_coco.yml) | COCO |
| `mask-rtdetr-hgnetv2-m-6x_640x640` | instance_segmentation | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/configs/mask_rtdetr/mask_rtdetr_hgnetv2_m_6x_coco.yml) | COCO |
| `mask-rtdetr-hgnetv2-s-6x_640x640` | instance_segmentation | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/configs/mask_rtdetr/mask_rtdetr_hgnetv2_s_6x_coco.yml) | COCO |
| `mask-rtdetr-hgnetv2-x-6x_640x640` | instance_segmentation | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/configs/mask_rtdetr/mask_rtdetr_hgnetv2_x_6x_coco.yml) | COCO |
| `yolact_regnet-x1.6gf_512x512` | instance_segmentation | MIT | MIT |  | [link](https://github.com/dbolya/yolact) | COCO |
| `yolact_regnet-x800mf_512x512` | instance_segmentation | MIT | MIT |  | [link](https://github.com/dbolya/yolact) | COCO |
| `yolo11-l-seg_640x640` | instance_segmentation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolo11-l-seg_640x640_pre-optimized` | instance_segmentation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolo11-m-seg_640x640` | instance_segmentation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolo11-m-seg_640x640_pre-optimized` | instance_segmentation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolo11-n-seg_640x640` | instance_segmentation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolo11-n-seg_640x640_pre-optimized` | instance_segmentation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolo11-s-seg_640x640` | instance_segmentation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolo11-s-seg_640x640_pre-optimized` | instance_segmentation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolo11-x-seg_640x640` | instance_segmentation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolo11-x-seg_640x640_pre-optimized` | instance_segmentation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolo26-l-seg_640x640` | instance_segmentation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolo26-m-seg_640x640` | instance_segmentation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolo26-n-seg_640x640` | instance_segmentation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolo26-s-seg_640x640` | instance_segmentation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolo26-x-seg_640x640` | instance_segmentation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolov12-seg-l_640x640` | instance_segmentation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/sunsmarterjie/yolov12) | COCO |
| `yolov12-seg-l_640x640_pre-optimized` | instance_segmentation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/sunsmarterjie/yolov12) | COCO |
| `yolov12-seg-m_640x640` | instance_segmentation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/sunsmarterjie/yolov12) | COCO |
| `yolov12-seg-m_640x640_pre-optimized` | instance_segmentation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/sunsmarterjie/yolov12) | COCO |
| `yolov12-seg-n_640x640` | instance_segmentation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/sunsmarterjie/yolov12) | COCO |
| `yolov12-seg-n_640x640_pre-optimized` | instance_segmentation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/sunsmarterjie/yolov12) | COCO |
| `yolov12-seg-s_640x640` | instance_segmentation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/sunsmarterjie/yolov12) | COCO |
| `yolov12-seg-s_640x640_pre-optimized` | instance_segmentation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/sunsmarterjie/yolov12) | COCO |
| `yolov12-seg-x_640x640` | instance_segmentation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/sunsmarterjie/yolov12) | COCO |
| `yolov12-seg-x_640x640_pre-optimized` | instance_segmentation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/sunsmarterjie/yolov12) | COCO |
| `yolov5-l-seg_640x640` | instance_segmentation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/yolov5) | COCO |
| `yolov5-m-seg_640x640` | instance_segmentation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/yolov5) | COCO |
| `yolov5-n-seg_640x640` | instance_segmentation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/yolov5) | COCO |
| `yolov5-s-seg_640x640` | instance_segmentation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/yolov5) | COCO |
| `yolov5-x-seg_640x640` | instance_segmentation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/yolov5) | COCO |
| `yolov8-l-seg_640x640` | instance_segmentation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolov8-l-seg_640x640_pre-optimized` | instance_segmentation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolov8-m-seg_640x640` | instance_segmentation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolov8-m-seg_640x640_pre-optimized` | instance_segmentation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolov8-n-seg_640x640` | instance_segmentation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolov8-n-seg_640x640_pre-optimized` | instance_segmentation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolov8-s-seg_640x640` | instance_segmentation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolov8-s-seg_640x640_pre-optimized` | instance_segmentation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolov8-x-seg_640x640` | instance_segmentation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolov8-x-seg_640x640_pre-optimized` | instance_segmentation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolov9-seg-c_640x640` | instance_segmentation | GPL-3.0 | GPL-3.0 | Copyleft | [link](https://github.com/WongKinYiu/yolov9) | COCO |
| `yolov9-seg-c_640x640_pre-optimized` | instance_segmentation | GPL-3.0 | GPL-3.0 | Copyleft | [link](https://github.com/WongKinYiu/yolov9) | COCO |
| `yolov9-seg-e_640x640` | instance_segmentation | GPL-3.0 | GPL-3.0 | Copyleft | [link](https://github.com/WongKinYiu/yolov9) | COCO |
| `yolov9-seg-e_640x640_pre-optimized` | instance_segmentation | GPL-3.0 | GPL-3.0 | Copyleft | [link](https://github.com/WongKinYiu/yolov9) | COCO |
| `yolov9-seg-gelan-c_640x640` | instance_segmentation | GPL-3.0 | GPL-3.0 | Copyleft | [link](https://github.com/WongKinYiu/yolov9) | COCO |
| `yolov9-seg-gelan-c_640x640_pre-optimized` | instance_segmentation | GPL-3.0 | GPL-3.0 | Copyleft | [link](https://github.com/WongKinYiu/yolov9) | COCO |
| `superpoint_480x640` | keypoint_detection | Magic Leap Custom License (Academic/Noncommercial) | Magic Leap Custom License (Academic/Noncommercial) | **Non-commercial use only** | [link](https://github.com/magicleap/SuperPointPretrainedNetwork) | HPatches |
| `zerodce-pp_400x600` | low_light_enhancement | CC-BY-NC-4.0 | CC-BY-NC-4.0 | **Non-commercial use only** | [link](https://github.com/Li-Chongyi/Zero-DCE_extension) | LOL |
| `zerodce_400x600` | low_light_enhancement | CC-BY-NC-4.0 | CC-BY-NC-4.0 | **Non-commercial use only** | [link](https://github.com/Li-Chongyi/Zero-DCE) | LOL |
| `pp-shituv1-mainbody-detection_640x640` | mainbody_detection | Apache-2.0 | Apache-2.0 |  | [link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/picodet_PPLCNet_x2_5_mainbody_lite_v1.0_infer.tar) | COCO |
| `pp-shituv2-mainbody-detection_640x640` | mainbody_detection | Apache-2.0 | Apache-2.0 |  | [link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/picodet_PPLCNet_x2_5_mainbody_lite_v1.0_infer.tar) | COCO |
| `damoyolo-l_640x640` | object_detection | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/tinyvision/DAMO-YOLO) | COCO |
| `damoyolo-m_640x640` | object_detection | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/tinyvision/DAMO-YOLO) | COCO |
| `damoyolo-s_640x640` | object_detection | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/tinyvision/DAMO-YOLO) | COCO |
| `damoyolo-t_640x640` | object_detection | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/tinyvision/DAMO-YOLO) | COCO |
| `damoyolo-tinynas-l20m_640x640` | object_detection | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/tinyvision/DAMO-YOLO) | COCO |
| `damoyolo-tinynas-l20t_640x640` | object_detection | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/tinyvision/DAMO-YOLO) | COCO |
| `damoyolo-tinynas-l25s_640x640` | object_detection | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/tinyvision/DAMO-YOLO) | COCO |
| `efficientdet-d1_640x640` | object_detection | LGPL-3.0 | LGPL-3.0 | Copyleft | [link](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) | COCO |
| `efficientdet-d2_768x768` | object_detection | LGPL-3.0 | LGPL-3.0 | Copyleft | [link](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) | COCO |
| `efficientdet-d4_1024x1024` | object_detection | LGPL-3.0 | LGPL-3.0 | Copyleft | [link](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) | COCO |
| `nanodet-plus-1.5x_416x416` | object_detection | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/RangiLyu/nanodet) | COCO |
| `nanodet-plus_416x416` | object_detection | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/RangiLyu/nanodet) | COCO |
| `nanodet-repvgg-a0_416x416` | object_detection | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/RangiLyu/nanodet) | COCO |
| `nanodet-repvgg-a12_640x640` | object_detection | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/Megvii-BaseDetection/YOLOX) | COCO |
| `nanodet-repvgg-a1_640x640` | object_detection | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/RangiLyu/nanodet) | COCO |
| `rtdetr-hgnetv2-h-6x_640x640` | object_detection | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/configs/rtdetr/rtdetr_hgnetv2_h_6x_coco.yml) | COCO |
| `rtdetr-hgnetv2-l-6x_640x640` | object_detection | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/configs/rtdetr/rtdetr_hgnetv2_l_6x_coco.yml) | COCO |
| `rtdetr-hgnetv2-x-6x_640x640` | object_detection | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/configs/rtdetr/rtdetr_hgnetv2_x_6x_coco.yml) | COCO |
| `rtdetr-r101vd-6x_640x640` | object_detection | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/configs/rtdetr/rtdetr_r101vd_6x_coco.yml) | COCO |
| `rtdetr-r18vd-6x_640x640` | object_detection | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/configs/rtdetr/rtdetr_r18vd_6x_coco.yml) | COCO |
| `rtdetr-r34vd-6x_640x640` | object_detection | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/configs/rtdetr/rtdetr_r34vd_6x_coco.yml) | COCO |
| `rtdetr-r50vd-6x_640x640` | object_detection | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/configs/rtdetr/rtdetr_r50vd_6x_coco.yml) | COCO |
| `rtdetr-r50vd-m-6x_640x640` | object_detection | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/configs/rtdetr/rtdetr_r50vd_m_6x_coco.yml) | COCO |
| `rtdetrv2-r101vd-6x_640x640` | object_detection | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/configs/rtdetrv2/rtdetrv2_r101vd_6x_coco.yml) | COCO |
| `rtdetrv2-r18vd-120e_640x640` | object_detection | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml) | COCO |
| `rtdetrv2-r34vd-120e_640x640` | object_detection | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/configs/rtdetrv2/rtdetrv2_r34vd_120e_coco.yml) | COCO |
| `rtdetrv2-r50vd-6x_640x640` | object_detection | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml) | COCO |
| `rtdetrv2-r50vd-m-7x_640x640` | object_detection | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/configs/rtdetrv2/rtdetrv2_r50vd_m_7x_coco.yml) | COCO |
| `rtdetrv3-r18vd-6x_640x640` | object_detection | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml) | COCO |
| `rtdetrv3-r50vd-6x_640x640` | object_detection | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/configs/rtdetrv3/rtdetrv3_r50vd_6x_coco.yml) | COCO |
| `ssd-mobilenetv1_300x300` | object_detection | MIT | MIT |  | [link](https://github.com/qfgaohao/pytorch-ssd) | PascalVOC2007 |
| `ssd-mobilenetv2-lite_300x300` | object_detection | MIT | MIT |  | [link](https://github.com/qfgaohao/pytorch-ssd) | PascalVOC2007 |
| `ssd-vgg16_300x300` | object_detection | MIT | MIT |  | [link](https://github.com/qfgaohao/pytorch-ssd) | PascalVOC2007 |
| `yolo11-l_640x640` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolo11-l_640x640_pre-optimized` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolo11-m_640x640` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolo11-m_640x640_pre-optimized` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolo11-n_640x640` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolo11-n_640x640_ppu` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolo11-n_640x640_pre-optimized` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolo11-s_640x640` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolo11-s_640x640_pre-optimized` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolo11-x_640x640` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolo11-x_640x640_pre-optimized` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolo26-l_640x640` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolo26-l_640x640_pre-optimized` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolo26-m_640x640` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolo26-m_640x640_pre-optimized` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolo26-n_640x640` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolo26-n_640x640_pre-optimized` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolo26-s_640x640` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolo26-s_640x640_pre-optimized` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolo26-x_640x640` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolo26-x_640x640_pre-optimized` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolov10-b_640x640` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/THU-MIG/yolov10) | COCO |
| `yolov10-l_640x640` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/THU-MIG/yolov10) | COCO |
| `yolov10-m_640x640` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/THU-MIG/yolov10) | COCO |
| `yolov10-n_640x640` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/THU-MIG/yolov10) | COCO |
| `yolov10-n_640x640_ppu` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/THU-MIG/yolov10) | COCO |
| `yolov10-s_640x640` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/THU-MIG/yolov10) | COCO |
| `yolov10-x_640x640` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/THU-MIG/yolov10) | COCO |
| `yolov12-l_640x640` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/sunsmarterjie/yolov12) | COCO |
| `yolov12-l_640x640_pre-optimized` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/sunsmarterjie/yolov12) | COCO |
| `yolov12-m_640x640` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/sunsmarterjie/yolov12) | COCO |
| `yolov12-m_640x640_pre-optimized` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/sunsmarterjie/yolov12) | COCO |
| `yolov12-n_640x640` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/sunsmarterjie/yolov12) | COCO |
| `yolov12-n_640x640_ppu` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/sunsmarterjie/yolov12) | COCO |
| `yolov12-n_640x640_pre-optimized` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/sunsmarterjie/yolov12) | COCO |
| `yolov12-s_640x640` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/sunsmarterjie/yolov12) | COCO |
| `yolov12-s_640x640_pre-optimized` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/sunsmarterjie/yolov12) | COCO |
| `yolov12-x_640x640` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/sunsmarterjie/yolov12) | COCO |
| `yolov12-x_640x640_pre-optimized` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/sunsmarterjie/yolov12) | COCO |
| `yolov13-l_640x640` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/iMoonLab/yolov13) | COCO |
| `yolov13-l_640x640_pre-optimized` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/iMoonLab/yolov13) | COCO |
| `yolov13-n_640x640` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/iMoonLab/yolov13) | COCO |
| `yolov13-n_640x640_pre-optimized` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/iMoonLab/yolov13) | COCO |
| `yolov13-s_640x640` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/iMoonLab/yolov13) | COCO |
| `yolov13-s_640x640_pre-optimized` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/iMoonLab/yolov13) | COCO |
| `yolov13-x_640x640` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/iMoonLab/yolov13) | COCO |
| `yolov13-x_640x640_pre-optimized` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/iMoonLab/yolov13) | COCO |
| `yolov3-gluon_416x416` | object_detection | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/dmlc/gluon-cv) | COCO |
| `yolov3-gluon_608x608` | object_detection | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/dmlc/gluon-cv) | COCO |
| `yolov3-tiny_416x416` | object_detection | YOLO License | YOLO License |  | [link](https://github.com/AlexeyAB/darknet) | COCO |
| `yolov3-tiny_416x416_ppu` | object_detection | YOLO License | YOLO License |  | [link](https://github.com/AlexeyAB/darknet) | COCO |
| `yolov3_416x416` | object_detection | YOLO License | YOLO License |  | [link](https://github.com/AlexeyAB/darknet) | COCO |
| `yolov3_416x416_ppu` | object_detection | YOLO License | YOLO License |  | [link](https://github.com/AlexeyAB/darknet) | COCO |
| `yolov3_608x608_ppu` | object_detection | YOLO License | YOLO License |  | [link](https://github.com/AlexeyAB/darknet) | COCO |
| `yolov3_640x640` | object_detection | YOLO License | YOLO License |  | [link](https://github.com/AlexeyAB/darknet) | COCO |
| `yolov4-leaky_512x512` | object_detection | YOLO License | YOLO License |  | [link](https://github.com/AlexeyAB/darknet) | COCO |
| `yolov4-t_416x416` | object_detection | YOLO License | YOLO License |  | [link](https://github.com/AlexeyAB/darknet) | COCO |
| `yolov4_512x512_ppu` | object_detection | YOLO License | YOLO License |  | [link](https://github.com/AlexeyAB/darknet) | COCO |
| `yolov5-l6_1280x1280` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/yolov5) | COCO |
| `yolov5-l_640x640` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/yolov5) | COCO |
| `yolov5-m6_1280x1280` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/yolov5) | COCO |
| `yolov5-m6_1280x1280_v6.1` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/yolov5) | COCO |
| `yolov5-m_640x640` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/yolov5) | COCO |
| `yolov5-m_640x640_nospp` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/yolov5/releases/tag/v2.0) | COCO |
| `yolov5-m_640x640_v6.1` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/yolov5) | COCO |
| `yolov5-n6_1280x1280` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/yolov5) | COCO |
| `yolov5-n6_1280x1280_v6.1` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/yolov5) | COCO |
| `yolov5-n_640x640` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/yolov5) | COCO |
| `yolov5-s6_1280x1280` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/yolov5) | COCO |
| `yolov5-s6_1280x1280_v6.1` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/yolov5) | COCO |
| `yolov5-s_320x320` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/yolov5) | COCO |
| `yolov5-s_640x640` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/yolov5) | COCO |
| `yolov5-s_640x640_c3tr` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/yolov5/tree/v6.0) | COCO |
| `yolov5-s_640x640_decode` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/yolov5) | COCO |
| `yolov5-s_640x640_nospp` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/yolov5/releases/tag/v2.0) | COCO |
| `yolov5-s_640x640_ppu` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/yolov5) | COCO |
| `yolov5-x6_1280x1280` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/yolov5) | COCO |
| `yolov5-x_640x640` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/yolov5) | COCO |
| `yolov5-xs_512x512_nospp` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/yolov5/releases/tag/v2.0) | COCO |
| `yolov6-l6_1280x1280` | object_detection | GPL-3.0 | GPL-3.0 | Copyleft | [link](https://github.com/meituan/YOLOv6) | COCO |
| `yolov6-l_640x640` | object_detection | GPL-3.0 | GPL-3.0 | Copyleft | [link](https://github.com/meituan/YOLOv6) | COCO |
| `yolov6-m6_1280x1280` | object_detection | GPL-3.0 | GPL-3.0 | Copyleft | [link](https://github.com/meituan/YOLOv6) | COCO |
| `yolov6-m_640x640` | object_detection | GPL-3.0 | GPL-3.0 | Copyleft | [link](https://github.com/meituan/YOLOv6) | COCO |
| `yolov6-n0_640x640_v0.1.0` | object_detection | GPL-3.0 | GPL-3.0 | Copyleft | [link](https://github.com/meituan/YOLOv6) | COCO |
| `yolov6-n0_640x640_v0.2.1` | object_detection | GPL-3.0 | GPL-3.0 | Copyleft | [link](https://github.com/meituan/YOLOv6) | COCO |
| `yolov6-n_640x640` | object_detection | GPL-3.0 | GPL-3.0 | Copyleft | [link](https://github.com/meituan/YOLOv6) | COCO |
| `yolov6-n_640x640_nmscore` | object_detection | GPL-3.0 | GPL-3.0 | Copyleft | [link](https://github.com/meituan/YOLOv6) | COCO |
| `yolov6-s_640x640` | object_detection | GPL-3.0 | GPL-3.0 | Copyleft | [link](https://github.com/meituan/YOLOv6) | COCO |
| `yolov7-d6_1280x1280` | object_detection | GPL-3.0 | GPL-3.0 | Copyleft | [link](https://github.com/WongKinYiu/yolov7) | COCO |
| `yolov7-e6_1280x1280` | object_detection | GPL-3.0 | GPL-3.0 | Copyleft | [link](https://github.com/WongKinYiu/yolov7) | COCO |
| `yolov7-e6e_1280x1280` | object_detection | GPL-3.0 | GPL-3.0 | Copyleft | [link](https://github.com/WongKinYiu/yolov7) | COCO |
| `yolov7-tiny_640x640` | object_detection | GPL-3.0 | GPL-3.0 | Copyleft | [link](https://github.com/WongKinYiu/yolov7) | COCO |
| `yolov7-w6_1280x1280` | object_detection | GPL-3.0 | GPL-3.0 | Copyleft | [link](https://github.com/WongKinYiu/yolov7) | COCO |
| `yolov7-w6_1280x1280_nodecode` | object_detection | GPL-3.0 | GPL-3.0 | Copyleft | [link](https://github.com/WongKinYiu/yolov7) | COCO |
| `yolov7-x_640x640` | object_detection | GPL-3.0 | GPL-3.0 | Copyleft | [link](https://github.com/WongKinYiu/yolov7) | COCO |
| `yolov7-x_640x640_ppu` | object_detection | GPL-3.0 | GPL-3.0 | Copyleft | [link](https://github.com/WongKinYiu/yolov7) | COCO |
| `yolov7_640x640` | object_detection | GPL-3.0 | GPL-3.0 | Copyleft | [link](https://github.com/WongKinYiu/yolov7) | COCO |
| `yolov7_640x640_nodecode` | object_detection | GPL-3.0 | GPL-3.0 | Copyleft | [link](https://github.com/WongKinYiu/yolov7) | COCO |
| `yolov7_640x640_ppu` | object_detection | GPL-3.0 | GPL-3.0 | Copyleft | [link](https://github.com/WongKinYiu/yolov7) | COCO |
| `yolov8-l_640x640` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolov8-l_640x640_pre-optimized` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolov8-m_640x640` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolov8-m_640x640_pre-optimized` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolov8-n_640x640` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolov8-n_640x640_ppu` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolov8-n_640x640_pre-optimized` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolov8-s_640x640` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolov8-s_640x640_decode` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolov8-s_640x640_ppu` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolov8-s_640x640_pre-optimized` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolov8-x_640x640` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolov8-x_640x640_pre-optimized` | object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCO |
| `yolov9-c_640x640` | object_detection | GPL-3.0 | GPL-3.0 | Copyleft | [link](https://github.com/WongKinYiu/yolov9) | COCO |
| `yolov9-c_640x640_pre-optimized` | object_detection | GPL-3.0 | GPL-3.0 | Copyleft | [link](https://github.com/WongKinYiu/yolov9) | COCO |
| `yolov9-e_640x640` | object_detection | GPL-3.0 | GPL-3.0 | Copyleft | [link](https://github.com/WongKinYiu/yolov9) | COCO |
| `yolov9-e_640x640_pre-optimized` | object_detection | GPL-3.0 | GPL-3.0 | Copyleft | [link](https://github.com/WongKinYiu/yolov9) | COCO |
| `yolov9-gelan-c_640x640` | object_detection | GPL-3.0 | GPL-3.0 | Copyleft | [link](https://github.com/WongKinYiu/yolov9) | COCO |
| `yolov9-gelan-c_640x640_pre-optimized` | object_detection | GPL-3.0 | GPL-3.0 | Copyleft | [link](https://github.com/WongKinYiu/yolov9) | COCO |
| `yolov9-m_640x640` | object_detection | GPL-3.0 | GPL-3.0 | Copyleft | [link](https://github.com/WongKinYiu/yolov9) | COCO |
| `yolov9-m_640x640_pre-optimized` | object_detection | GPL-3.0 | GPL-3.0 | Copyleft | [link](https://github.com/WongKinYiu/yolov9) | COCO |
| `yolov9-s_640x640` | object_detection | GPL-3.0 | GPL-3.0 | Copyleft | [link](https://github.com/WongKinYiu/yolov9) | COCO |
| `yolov9-s_640x640_pre-optimized` | object_detection | GPL-3.0 | GPL-3.0 | Copyleft | [link](https://github.com/WongKinYiu/yolov9) | COCO |
| `yolov9-t_640x640` | object_detection | GPL-3.0 | GPL-3.0 | Copyleft | [link](https://github.com/WongKinYiu/yolov9) | COCO |
| `yolov9-t_640x640_ppu` | object_detection | GPL-3.0 | GPL-3.0 | Copyleft | [link](https://github.com/WongKinYiu/yolov9) | COCO |
| `yolov9-t_640x640_pre-optimized` | object_detection | GPL-3.0 | GPL-3.0 | Copyleft | [link](https://github.com/WongKinYiu/yolov9) | COCO |
| `yolox-l-leaky_640x640` | object_detection | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/Megvii-BaseDetection/YOLOX) | COCO |
| `yolox-l_640x640` | object_detection | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/Megvii-BaseDetection/YOLOX) | COCO |
| `yolox-m_640x640` | object_detection | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/Megvii-BaseDetection/YOLOX) | COCO |
| `yolox-s-leaky_640x640` | object_detection | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/Megvii-BaseDetection/YOLOX) | COCO |
| `yolox-s-wide-leaky_640x640` | object_detection | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/Megvii-BaseDetection/YOLOX) | COCO |
| `yolox-s_640x640` | object_detection | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/Megvii-BaseDetection/YOLOX) | COCO |
| `yolox-s_640x640_ppu` | object_detection | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/Megvii-BaseDetection/YOLOX) | COCO |
| `yolox-t_416x416` | object_detection | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/Megvii-BaseDetection/YOLOX) | COCO |
| `yolox-x_640x640` | object_detection | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/Megvii-BaseDetection/YOLOX) | COCO |
| `dope-hope-ketchup_480x640` | object_pose_estimation | NVIDIA Source Code License-NC | NVIDIA Source Code License-NC | **Non-commercial use only** | [link](https://github.com/NVlabs/Deep_Object_Pose) | HOPE |
| `yolo26-l-obb_1024x1024` | oriented_object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | DOTAv1 |
| `yolo26-m-obb_1024x1024` | oriented_object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | DOTAv1 |
| `yolo26-n-obb_1024x1024` | oriented_object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | DOTAv1 |
| `yolo26-s-obb_1024x1024` | oriented_object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | DOTAv1 |
| `yolo26-x-obb_1024x1024` | oriented_object_detection | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | DOTAv1 |
| `yolopv2_384x640` | panoptic_driving_perception | MIT | MIT |  | [link](https://github.com/CAIC-AD/YOLOPv2) | BDD100K |
| `deepmar_resnet18_224x224` | person_attribute | Custom License (Academic Research Only) | Custom License (Academic Research Only) | **Non-commercial use only** | [link](https://github.com/dangweili/pedestrian-attribute-recognition-pytorch) | PETA |
| `deepmar_resnet50_224x224` | person_attribute | Custom License (Academic Research Only) | Custom License (Academic Research Only) | **Non-commercial use only** | [link](https://github.com/dangweili/pedestrian-attribute-recognition-pytorch) | PETA |
| `repvgg-a0-reid_256x128` | person_reid | MIT | MIT |  | [link](https://github.com/DingXiaoH/RepVGG) | Market1501 |
| `centerpose-fpn_regnet-x1.6gf_512x512` | pose_estimation | MIT | MIT |  | [link](https://github.com/tensorboy/centerpose) | COCOPose |
| `centerpose_regnet-x800mf_640x640` | pose_estimation | MIT | MIT |  | [link](https://github.com/tensorboy/centerpose) | COCOPose |
| `centerpose_repvgg-a0_416x416` | pose_estimation | MIT | MIT |  | [link](https://github.com/tensorboy/centerpose) | COCOPose |
| `dark-hrnet-w32_192x256` | pose_estimation | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.6/configs/pphuman/dark_hrnet_w32_256x192.yml) | COCOPoseTopDown |
| `vitpose-s_256x192` | pose_estimation | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/ViTAE-Transformer/ViTPose) | COCOPoseTopDown |
| `yolo11-l-pose_640x640` | pose_estimation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCOPose |
| `yolo11-l-pose_640x640_pre-optimized` | pose_estimation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCOPose |
| `yolo11-m-pose_640x640` | pose_estimation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCOPose |
| `yolo11-m-pose_640x640_pre-optimized` | pose_estimation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCOPose |
| `yolo11-n-pose_640x640` | pose_estimation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCOPose |
| `yolo11-n-pose_640x640_pre-optimized` | pose_estimation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCOPose |
| `yolo11-s-pose_640x640` | pose_estimation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCOPose |
| `yolo11-s-pose_640x640_pre-optimized` | pose_estimation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCOPose |
| `yolo11-x-pose_640x640` | pose_estimation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCOPose |
| `yolo11-x-pose_640x640_pre-optimized` | pose_estimation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCOPose |
| `yolo26-l-pose_640x640` | pose_estimation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCOPose |
| `yolo26-l-pose_640x640_pre-optimized` | pose_estimation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCOPose |
| `yolo26-m-pose_640x640` | pose_estimation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCOPose |
| `yolo26-m-pose_640x640_pre-optimized` | pose_estimation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCOPose |
| `yolo26-n-pose_640x640` | pose_estimation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCOPose |
| `yolo26-n-pose_640x640_pre-optimized` | pose_estimation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCOPose |
| `yolo26-s-pose_640x640` | pose_estimation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCOPose |
| `yolo26-s-pose_640x640_pre-optimized` | pose_estimation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCOPose |
| `yolo26-x-pose_640x640` | pose_estimation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCOPose |
| `yolo26-x-pose_640x640_pre-optimized` | pose_estimation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCOPose |
| `yolov5-s6-pose_640x640` | pose_estimation | GPL-3.0 | GPL-3.0 | Copyleft | [link](https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose) | COCOPose |
| `yolov8-l-pose_640x640` | pose_estimation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCOPose |
| `yolov8-l-pose_640x640_pre-optimized` | pose_estimation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCOPose |
| `yolov8-m-pose_640x640` | pose_estimation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCOPose |
| `yolov8-m-pose_640x640_pre-optimized` | pose_estimation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCOPose |
| `yolov8-n-pose_640x640` | pose_estimation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCOPose |
| `yolov8-n-pose_640x640_pre-optimized` | pose_estimation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCOPose |
| `yolov8-s-pose_640x640` | pose_estimation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCOPose |
| `yolov8-s-pose_640x640_pre-optimized` | pose_estimation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCOPose |
| `yolov8-x-pose_640x640` | pose_estimation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCOPose |
| `yolov8-x-pose_640x640_pre-optimized` | pose_estimation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/ultralytics/ultralytics) | COCOPose |
| `bisenetv1_1024x2048` | semantic_segmentation | MIT | MIT |  | [link](https://github.com/CoinCheung/BiSeNet) | Cityscapes |
| `bisenetv2_1024x2048` | semantic_segmentation | MIT | MIT |  | [link](https://github.com/CoinCheung/BiSeNet) | Cityscapes |
| `casvit-t-fpn-resnet50_512x512` | semantic_segmentation | MIT | MIT |  | [link](https://github.com/Tianfang-Zhang/CAS-ViT) | ADE20K |
| `ddrnet23-slim_1024x2048` | semantic_segmentation | MIT | MIT |  | [link](https://github.com/ydhongHIT/DDRNet) | Cityscapes |
| `ddrnet23_1024x2048` | semantic_segmentation | MIT | MIT |  | [link](https://github.com/ydhongHIT/DDRNet) | Cityscapes |
| `deeplabv3-resnet101_512x512` | semantic_segmentation | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | PascalVOC2012 |
| `deeplabv3-resnet50_512x512` | semantic_segmentation | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | PascalVOC2012 |
| `deeplabv3_mobilenetv2_512x512` | semantic_segmentation | MIT | MIT |  | [link](https://github.com/VainF/DeepLabV3Plus-Pytorch) | PascalVOC2012 |
| `deeplabv3_mobilenetv2_513x513_nodilation` | semantic_segmentation | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/tensorflow/models/tree/master/research/deeplab) | PascalVOC2012 |
| `deeplabv3plus-drn_512x512` | semantic_segmentation | MIT | MIT |  | [link](https://github.com/jfzhang95/pytorch-deeplab-xception) | PascalVOC2012 |
| `deeplabv3plus-resnet101_512x512` | semantic_segmentation | MIT | MIT |  | [link](https://github.com/VainF/DeepLabV3Plus-Pytorch) | PascalVOC2012 |
| `deeplabv3plus-resnet50_512x512` | semantic_segmentation | MIT | MIT |  | [link](https://github.com/VainF/DeepLabV3Plus-Pytorch) | PascalVOC2012 |
| `deeplabv3plus_mobilenetv1_512x512` | semantic_segmentation | MIT | MIT |  | [link](https://github.com/jfzhang95/pytorch-deeplab-xception) | PascalVOC2012 |
| `deeplabv3plus_mobilenetv2_512x512` | semantic_segmentation | MIT | MIT |  | [link](https://github.com/VainF/DeepLabV3Plus-Pytorch) | PascalVOC2012 |
| `fcn8_resnet18_1024x1920` | semantic_segmentation | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/open-mmlab/mmsegmentation) | Cityscapes |
| `fcn8_resnet50_512x512` | semantic_segmentation | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/pytorch/vision) | PascalVOC2012 |
| `pidnet-s_1024x2048` | semantic_segmentation | MIT | MIT |  | [link](https://github.com/XuJiacong/PIDNet) | Cityscapes |
| `pp-liteseg-stdc1-camvid-10k_720x960` | semantic_segmentation | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.8/configs/pp_liteseg/pp_liteseg_stdc1_camvid_960x720_10k.yml) | CamVid |
| `segformer_mit-b0_512x1024` | semantic_segmentation | NVIDIA Source Code License-NC | NVIDIA Source Code License-NC | **Non-commercial use only** | [link](https://github.com/NVlabs/SegFormer) | Cityscapes |
| `stdc1-seg50_512x1024` | semantic_segmentation | MIT | MIT |  | [link](https://github.com/MichaelFan01/STDC-Seg) | Cityscapes |
| `stdc1-seg75_768x1536` | semantic_segmentation | MIT | MIT |  | [link](https://github.com/MichaelFan01/STDC-Seg) | Cityscapes |
| `stdc2-seg50_512x1024` | semantic_segmentation | MIT | MIT |  | [link](https://github.com/MichaelFan01/STDC-Seg) | Cityscapes |
| `stdc2-seg75_768x1536` | semantic_segmentation | MIT | MIT |  | [link](https://github.com/MichaelFan01/STDC-Seg) | Cityscapes |
| `unet_mobilenetv2_256x256` | semantic_segmentation | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/tensorflow/examples) | OxfordIIITPet |
| `espcn-x2_17x17` | super_resolution | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/Lornatang/ESPCN-PyTorch) | BSD100 |
| `espcn-x3_17x17` | super_resolution | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/Lornatang/ESPCN-PyTorch) | BSD100 |
| `espcn-x4_17x17` | super_resolution | Apache-2.0 | Apache-2.0 |  | [link](https://github.com/Lornatang/ESPCN-PyTorch) | BSD100 |
| `realesrgan-x2_192x192` | super_resolution | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/xinntao/Real-ESRGAN) | BSD100 |
| `realesrgan-x4_192x192` | super_resolution | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/xinntao/Real-ESRGAN) | BSD100 |
| `realesrgan-x8_192x192` | super_resolution | BSD-3-Clause | BSD-3-Clause |  | [link](https://github.com/ai-forever/Real-ESRGAN) | BSD100 |
| `eigenplaces-resnet18_512x512` | visual_place_recognition | MIT | MIT |  | [link](https://github.com/gmberton/EigenPlaces) | HPatches |
| `eigenplaces-resnet50_512x512` | visual_place_recognition | MIT | MIT |  | [link](https://github.com/gmberton/EigenPlaces) | HPatches |
| `pp-shituv2-feature-extraction_224x224` | visual_place_recognition | Apache-2.0 | Apache-2.0 |  | [link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/PP-ShiTuV2/general_PPLCNetV2_base_pretrained_v1.0_infer.tar) | HPatches |
| `clip-img_resnet50_224x224_openai` | zero_shot_image_classification | MIT | MIT |  | [link](https://github.com/mlfoundations/open_clip) | ILSVRC2012 |
| `clip-img_resnet50x16_384x384_openai-wit` | zero_shot_image_classification | MIT | MIT |  | [link](https://github.com/mlfoundations/open_clip) | ILSVRC2012 |
| `clip-img_vit-b16-quickgelu_224x224_metaclip-fullcc` | zero_shot_image_classification | CC-BY-NC-4.0 | MIT | **Non-commercial use only** | [link](https://github.com/mlfoundations/open_clip) | ILSVRC2012 |
| `clip-img_vit-b32_256x256_datacomp-s34b-b86k` | zero_shot_image_classification | MIT | MIT |  | [link](https://github.com/mlfoundations/open_clip) | ILSVRC2012 |
| `clip-img_vit-l14-quickgelu_224x224_dfn2b` | zero_shot_image_classification | Apple Machine Learning Research Model License | MIT | **Non-commercial use only** | [link](https://github.com/mlfoundations/open_clip) | ILSVRC2012 |
| `clip-img_vit-l14_224x224_datacomp-xl-s13b-b90k` | zero_shot_image_classification | MIT | MIT |  | [link](https://github.com/mlfoundations/open_clip) | ILSVRC2012 |
| `fastsam-s_1024x1024` | zero_shot_instance_segmentation | AGPL-3.0 | AGPL-3.0 | Copyleft | [link](https://github.com/CASIA-IVA-Lab/FastSAM) | COCO |

## Evaluation datasets

Datasets are not redistributed. Check the terms of each dataset before using it.

| Dataset | License / terms | Terms |
|---|---|---|
| ADE20K | ADE20K Terms of Use (non-commercial); annotations: BSD-3-Clause | [link](https://ade20k.csail.mit.edu/terms/) |
| AFLW20003D | Not specified | [link](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm) |
| BDD100K | BDD100K License (non-commercial) | [link](https://doc.bdd100k.com/license.html) |
| BSD100 | BSDS Terms (non-commercial research) | [link](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) |
| BSD68 | BSDS Terms (non-commercial research) | [link](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) |
| CBSD68 | BSDS Terms (non-commercial research) | [link](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) |
| COCO | CC-BY-4.0 (annotations); images: Flickr Terms of Use | [link](https://cocodataset.org/#termsofuse) |
| COCOPose | CC-BY-4.0 (annotations); images: Flickr Terms of Use | [link](https://cocodataset.org/#termsofuse) |
| COCOPoseTopDown | CC-BY-4.0 (annotations); images: Flickr Terms of Use | [link](https://cocodataset.org/#termsofuse) |
| CamVid | Not specified | [link](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) |
| CelebA | CelebA Agreement (non-commercial research) | [link](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) |
| Cityscapes | Cityscapes License (non-commercial) | [link](https://www.cityscapes-dataset.com/license/) |
| DOTAv1 | DOTA Terms (academic use only) | [link](https://captain-whu.github.io/DOTA/dataset.html) |
| Distinctions646 | Distinctions-646 Terms (non-commercial, on request) | [link](https://github.com/wukaoliu/CVPR2020-HAttMatting) |
| HOPE | CC-BY-NC-SA-4.0 | [link](https://github.com/swtyree/hope-dataset) |
| HPatches | MIT (images from mixed sources) | [link](https://github.com/hpatches/hpatches-dataset) |
| HandKeypoints | CC-BY-NC-SA-4.0 | [link](https://docs.ultralytics.com/datasets/pose/hand-keypoints/) |
| HandKeypointsDetection | CC-BY-NC-SA-4.0 | [link](https://docs.ultralytics.com/datasets/pose/hand-keypoints/) |
| ILSVRC2012 | ImageNet Terms of Access (non-commercial research) | [link](https://image-net.org/accessagreement) |
| ImageNetText | ImageNet Terms of Access (non-commercial research) | [link](https://image-net.org/accessagreement) |
| KITTI | CC-BY-NC-SA-3.0 | [link](https://www.cvlibs.net/datasets/kitti/) |
| LFW | Not specified | [link](http://vis-www.cs.umass.edu/lfw/) |
| LOL | Not specified | [link](https://daooshee.github.io/BMVC2018website/) |
| MVTecAD | CC-BY-NC-SA-4.0 | [link](https://www.mvtec.com/company/research/datasets/mvtec-ad) |
| Market1501 | Market-1501 Terms (research only) | [link](https://zheng-lab-anu.github.io/Project/project_reid.html) |
| NYUDepthv2 | Not specified | [link](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) |
| OxfordIIITPet | CC-BY-SA-4.0 | [link](https://www.robots.ox.ac.uk/~vgg/data/pets/) |
| PETA | PETA Terms (research only) | [link](https://mmlab.ie.cuhk.edu.hk/projects/PETA.html) |
| PascalVOC2007 | Flickr Terms of Use (images) | [link](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/) |
| PascalVOC2012 | Flickr Terms of Use (images) | [link](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) |
| WiderFace | Non-commercial research only | [link](http://shuoyang1213.me/WIDERFACE/) |
