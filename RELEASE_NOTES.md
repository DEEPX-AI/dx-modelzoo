# RELEASE_NOTES

## v0.9.0 / 2026-03-30

### 1. Changed
- YOLOXS_PPU preprocessing changed to letterbox (pad) mode

### 2. Fixed
- Fixed IndexError in YOLOXS_PPU PPU postprocessing caused by invalid layer_idx in hardware output
- **MobileViT**, **FastViT**, **ViT**, **EfficientNet**

### 3. Added
- **Detection**:
  - CenterNet: CenterNet_ResNet18, CenterNet_ResNet50
  - DamoYolo: DamoYoloL
  - EfficientDet: EfficientDet_D0, EfficientDet_D2, EfficientDet_D3, EfficientDet_D4, EfficientDet_D5, EfficientDet_D6
  - NanoDet: NanoDet_Plus, NanoDet_Plus_15
  - YoloV3: YoloV3_Gluon_416, YoloV3_Gluon_608, YOLOV3_416_PPU, YOLOV3_608_PPU, YOLOV3Tiny_PPU
  - YoloV4: YOLOV4_PPU
  - YoloV5: YoloV5L_640, YoloV5L6_1280, YoloV5M_640, YoloV5M6_1280, YoloV5M6_1, YoloV5M6_61_1280, YoloV5M_WoSpp_640, YoloV5N6_1280, YoloV5N6_61_1280, YoloV5S_320, YoloV5S_640, YoloV5S6_1280, YoloV5S6_61_1280, YoloV5S_BboxDecoding_640, YoloV5S_C3tr_640, YoloV5S_WoSpp_640, YoloV5XS_WoSpp_512, YoloV5X_640, YoloV5X6_1280, YOLOV5S_PPU
  - YoloV6: YoloV6L_640, YoloV6M_640, YoloV6N0_1_0, YoloV6N0_2_1, YoloV6N6_1280, YoloV6N_NmsCore_640, YoloV6S_640, YoloV6S6_1280
  - YoloV7: YoloV7D6_1280, YoloV7E6E_1280, YoloV7_W6, YoloV7_W6_wo_decoding, YoloV7_wo_decoding, YoloV7_X, YOLOV7_PPU, YOLOV7X_PPU
  - YoloV8: YoloV8S_decoding, YOLOV8N_PPU, YOLOV8S_PPU
  - YoloV9: YoloV9_GELAN_C, YoloV9M, YOLOV9T_PPU
  - YoloV10: YoloV10N_PPU
  - YoloX: YoloXL_640, YoloXM_640, YoloXN_416, YoloXTiny_416, YoloXX_640, YOLOXS_PPU
  - Yolo26:
  -   OBB: YOLO26n_OBB, YOLO26s_OBB, YOLO26m_OBB, YOLO26l_OBB, YOLO26x_OBB
  -   Seg: Yolo26N_Seg, Yolo26S_Seg, Yolo26M_Seg, Yolo26L_Seg, Yolo26X_Seg
  -   Pose: Yolo26N_Pose, Yolo26S_Pose, Yolo26M_Pose, Yolo26L_Pose, Yolo26X_Pose
  - tinynas: tinynas
  - Others: SSDVGG16, YOLO_DeepX_640, ResNet18_BRECQ, ResNeXt50_32x4d_imgclsmob

- **Classification**:
  - FastViT: FastViT
  - RepGhost: RepGhost
  - Yolo26-cls: YOLO26n-cls, YOLO26s-cls, YOLO26m-cls, YOLO26l-cls, YOLO26x-cls

- **Segmentation**:
  - Yolo26: Yolo26N_Seg, Yolo26S_Seg, Yolo26M_Seg, Yolo26L_Seg, Yolo26X_Seg

- **Pose**:
  - Yolo26: Yolo26N_Pose, Yolo26S_Pose, Yolo26M_Pose, Yolo26L_Pose, Yolo26X_Pose

## DX-MODELZOO v0.7.0 / 2026-01-20

### 1. Changed
- **Environment Variable Update:** Added `DXNN_DEVICES` for device configuration
- **Evaluator Improvements:** 
  - Refactored log format for better readability
  - Applied async run to all task evaluators for improved performance
  - Enhanced progress bar to include dataset name

### 2. Fixed
- **Data Type Handling:** Ensured input data type matches expected dtype in DxRuntimeSession run methods
- **Model-Specific Fixes:**
  - Added missing normalization and division steps in GoogLeNet NPU preprocessing
  - Corrected keypoint scaling logic in COCO pose evaluator
  - Removed redundant preprocessing steps from MobileViT, ViT, FastViT, and EfficientNet models
- **Detection & NMS:** Enhanced box processing, NMS conversion, and COCO evaluation options
- **Resize Logic:** Adjusted padding logic in PadResize class for better handling of pad_location
- **Import Path:** Fixed import path for object_detection module

### 3. Added
This release includes a major expansion of **dx-modelzoo v0.7.0**, adding **85+ new models** across multiple computer vision tasks.

#### **Compile Command**
- Added `compile` command for generating `.dxnn` files directly from ONNX and JSON files
- Support for GPU option in quantization during compilation

#### **New Model Categories & Architectures**

##### **Object Detection**
- **YOLO26 Series:** yolo26n, yolo26s, yolo26m, yolo26l, yolo26x
- **YOLOv3 Series:** YoloV3_416, YoloV3_Tiny

##### **Semantic Segmentation**
- **DeepLab Series:** DeepLabV3PlusMobileNetV2
- **FCN Series:** FCN8_ResNet50, FCN8_ResNet18

##### **Image Classification**
- **FastViT Series:** FastViT_MA36, FastViT_SA24, FastViT_SA36
- **EfficientFormer Series:** EfficientFormer_L3, EfficientFormer_L7
- **DenseNet Series:** DenseNet169, DenseNet201
- **EfficientNet Series:** EfficientNetV2L
- **RegNet Series:** RegNetX_1_6GF, RegNetX_1_6GF_3, RegNetX_16GF, RegNetX_3_2GF, RegNetX_32GF, RegNetX_8GF
- **ResNeXt Series:** ResNeXt101_64x4d
- **RepGhost Series:** RepGhost_2_0x
- **ShuffleNet Series:** ShuffleNetV1_x1_0, ShuffleNetV2_x0_5, ShuffleNetV2_x1_0, ShuffleNetV2_x1_5, ShuffleNetV2_x2_0
- **VGG Series:** VGG16, VGG16BN, VGG19
- **MnasNet Series:** MnasNet0_5, MnasNet0_75, MnasNet1_0, MnasNet1_3
- **Others:** 
  - GhostNet
  - SqueezeNet1_3

#### **Infrastructure Improvements**
- **Session Management:** Added environment variable (`DXNN_DEVICES`) for device configuration
- **Async Evaluation:** Full async support across all evaluators for better performance
- **Installation:** Enhanced install.sh to support dx_rt dependency installation with `--all` flag

---

## DX-MODELZOO v0.6.0 / 2025-12-29

### 1. Changed

### 2. Fixed

### 3. Added
- We are excited to announce a major update to **dx-modelzoo**, significantly expanding our evaluation capabilities. The number of supported models for evaluation has increased from **59 to 110**.

#### Key Highlights
* **Scale Increase:** Expanded evaluation support to a total of 110 models (51 new models added).
* **Latest Architectures:** Integration of state-of-the-art models including **YOLOv10** and **YOLOv11**.
* **Broadened Coverage:** Enhanced support for various tasks including Object Detection, Classification, Segmentation, and Denoising.

#### New Supported Model List

The following models are now available for evaluation:

##### **Object Detection & Face Detection**
* **YOLOv11 Series:** YOLOV11N, YOLOV11S, YOLOV11M, YOLOV11L, YOLOV11X
* **YOLOv10 Series:** YOLOV10N, YOLOV10S, YOLOV10M, YOLOV10B, YOLOV10L, YOLOV10X
* **DamoYolo Series:** DamoYoloT (1, 2), DamoYoloS (1, 2), DamoYoloM (1, 2), DamoYoloL-1
* **Others:** NanoDet_RepVGG, NanoDet_RepVGGA1, SCRFD (500M, 2.5G, 10G), SSDMV1, SSDMV2Lite

##### **Classification**
* **ResNet & Variants:** ResNet (18, 34, 50, 101), ResNeXt (26, 50), WideResNet (50, 101)
* **MobileNet & EfficientNet:** MobileNet (V1, V2, V3L), EfficientNet (B2, V2S), EfficientNet_Lite (0, 1, 2, 3, 4)
* **RegNet Series:** RegNetX (400MF, 800MF), RegNetY (200MF, 400MF, 800MF)
* **VGG & SqueezeNet:** VGG (11, 11BN, 13, 13BN, 19BN), SqueezeNet (1.0, 1.1)
* **Others:** InceptionV1, HarDNet (39DS, 68), RepVGGA1, RepVGGA2, OSNet (0.25, 0.5)

##### **Other Tasks**
* **Segmentation:** DeepLabV3PlusMobilenet
* **Denoising:** DnCNN (15, 25, 50)
* **Depth Estimation:** FastDepth

---

## DX-MODELZOO v0.5.1 / 2025-08-18

### 1. Changed
- None

### 2. Fixed
- Enhance Python installation script to use minimum required version if no specific version is provided
  - Reorder venv force remove logic in install_python_and_venv function
  - Refactor Python installation script to streamline checks

### 3. Added
- None

---

## DX-MODELZOO v0.5.0 / 2025-08-18

### 1. Changed
- Add --symlink_target_path option to install_python_and_venv.sh
  - Add validation and error handling for symlink operations
- Consolidate all setup functionality into setup.sh and install_python_and_venv.sh
  - Enable venv creation at target path with symlink at specified path
- Fix setup.sh parameter parsing and forwarding for symlink options
- Remove deprecated scripts: install.sh, setup_venv.sh
- Update help documentation with symlink usage examples

### 2. Fixed
- remove temporary workaround code  from postprocessing
  - Fixed output tensor order mismatch in DXNN (dx_com v2.0.0)
  - Removed temporary workaround wrapper functions

### 3. Added

---

## DX-MODELZOO v0.4.0 / 2025-07-28

### 1. Changed
- feat: Refactor dxmz setup and update dx_com version
  - [setup.sh] Adds color-coded logging and common utility scripts.
- docs: Improve documentation for benchmark command (remove releated content)
  - This commit updates the Benchmark documentation, correcting a typo and removing the integrated benchmarking section. The integrated benchmarking section was removed(temporary).
- docs: Update Docker installation instructions in README.md
- chore: remove Performance data table in 'docs/source/index.md'

### 2. Fixed
- fix: Differentiate venv paths for local and container installations
  - When  is executed for a local (host) installation after a Docker-based setup has already created a virtual environment, a conflict can arise. Both installation methods would attempt to use the same venv path (), potentially leading to a corrupted environment.

### 3. Added

---

## DX-MODELZOO v0.3.1 / 2025-06-12
### 1. Changed
- None
### 2. Fixed
- chore: fix duplicated parsing error in 'yolov7_face_postprocessing_wrapper()'
- fix:  Temporary workaround for the mismatch in output tensor order between the original ONNX model and DXNN
### 3. Added
- chore: support '--debug' flag in 'dxmz eval' to enable detailed logs

---

## DX-MODELZOO v0.1.5 / 2025-06-04
### 1. Changed
- None
### 2. Fixed
- Fix error for Image Denoising Add 'bsd68' EvaluationType in enums.py
### 3. Added
- Add internal scripts (log parser, run eval onnx and so on)

---

## DX-MODELZOO v0.1.3 / 2025-05-26
### 1. Changed
- support check and install 'jq' and ' curl' in download_onnx_json_from*.sh
- support '--exclude_from', '--include_from' and '--get_model_list' options
### 2. Fixed
- fix problem for SSL error using curl on intranet
### 3. Added
- None

---

## DX-MODELZOO v0.1.2 / 2025-05-15
### 1. Changed
- None
### 2. Fixed
- fix: Support running setup.sh when using modelzoo standalone (without DX-AS all suite)
- fix: Check Python venv setup and activation status in install.sh and handle errors accordingly
### 3. Added
- None

---

## DX-MODELZOO v0.1.1 - 2025-05-14

### Fixes

- dxmz eval args validation updated

### Miscellaneous Tasks

- add download dataset & model & auto compile scripts
- update internal excluded
- update internal scripts
- add evaluation scripts
- add internal shell scripts

---

## DX-MODELZOO v0.1.0 - 2025-05-12

### Features

- *(__init__.py)* add arguments
- *(models,-dataset,-evaluator)* create models, dataset, evaluator ABC
- *(preprocessing)* create preprocessings
- *(session)* create Session
- *(imagenet.py)* create ImagenetDataset
- *(ic_evaluator.py)* create ICEvaluator
- create user command and add ResNet
- *(dx_runtime_session.py)* create DxRuntimSession
- Add YOLO
- *(dx_runtime_session.py)* do not except transpose preprocessing when using npu
- add DeepLabV3PlusMobilenet
- add metric, perfromances to model info
- *(resnet)* add resnet models
- *(model_dict.py)* update model_dict
- add yolov5 models
- add deeplabv3plus models
- add resnext models
- add Regnet models
- add densenet models
- add Efficientnet models
- add hardnet model
- add mobilenet models
- add squeezenet models
- add vgg models
- add WideResnet models
- add alexnet
- add vgg models
- add mobilenetv3 models
- add osnet and repvgg models
- *(resize.py)* add pycls mode
- add resize default mode
- add bicubic mode
- add od models
- *(ssd.py)* create ssd models
- add model to model_dict
- *(dncnn)* add denosing models
- *(bisenet.py)* add bisenet.py
- add yolov5 face
- add yolov7 face models
- *(yolo.py)* add yolov8 models
- add ./tools for generation of benchmark report; feat: add ./docs and mkdocs.yml for user guide document; docs: add user guide in  README.md
- add run benchmark to dxmz
- add internal benchmarking for internal experiments
- add prefect to handle multi-threading
- merge internal and public evaluation code.
- Add FPS in evaluation

### Fixes

- *(preprocessing)* fix parse_npu_preprocessing_ops bugs
- *(preprocessing)* remove transpose from npu preprocessings
- *(SegmentationEvaluator)* fix segmentation evaluator bugs
- fix model_dict class name

### Refactor

- refactoring base classes's interface
- change nms using torch
- add setup script for alpine docker image
- update python installation & dxas path

### Documentation

- add docs
- add missing docs
- clean up 'mkdocs' documant

### Testing

- add test of PreprocessingCompose

### Miscellaneous Tasks

- add DxRuntime session type
- missing command
- change model name
- change model name
- chaneg model_dict
- modify the output json structure; update the performance value in modelzoo table.
- modify the table format in user guide
- initial commit on github actions and required files
- fix typos
- modify readme
- modify readme
- update .github/workflows/public-main.yml
- update .github/workflows/release-please.yml
- update attributes & release excluded
- update .github/release-excluded

### build

- *(setup.py)* create sutup.py
- *(setpu.py)* add install requires
- *(setup.py)* update install requires

### debug

- change the log output.
- add a config file to handle setting
- modify files for  commend line.
- add condition for yolov7_face outputs

<!-- generated by git-cliff -->
