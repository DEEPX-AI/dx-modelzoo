# RELEASE_NOTES

## v0.4.0 / 2025-08-18

### 1. Changed
- Add --symlink_target_path option to install_python_and_venv.sh
  - Add validation and error handling for symlink operations
- Consolidate all setup functionality into setup.sh and install_python_and_venv.sh
  - Enable venv creation at target path with symlink at specified path
- Fix setup.sh parameter parsing and forwarding for symlink options
- Remove deprecated scripts: install.sh, setup_venv.sh
- Update help documentation with symlink usage examples

### 2. Fixed

### 3. Added

## v0.4.0 / 2025-08-13

### 1. Changed
  - [setup.sh] Adds color-coded logging and common utility scripts.
- chore: remove Performance data table in 'docs/source/index.md'
- docs: Improve documentation for benchmark command (remove releated content)
- docs: Update Docker installation instructions in README.md
- feat: Refactor dxmz setup and update dx_com version
  - This commit updates the Benchmark documentation, correcting a typo and removing the integrated benchmarking section. The integrated benchmarking section was removed(temporary).

### 2. Fixed
- fix: Differentiate venv paths for local and container installations
  - When  is executed for a local (host) installation after a Docker-based setup has already created a virtual environment, a conflict can arise. Both installation methods would attempt to use the same venv path (), potentially leading to a corrupted environment.
    **Container Mode Path:**
    **Local Mode Path:**

### 3. Added

## v0.3.1 / 2025-06-12
### 1. Changed
- None
### 2. Fixed
- chore: fix duplicated parsing error in 'yolov7_face_postprocessing_wrapper()'
- fix:  Temporary workaround for the mismatch in output tensor order between the original ONNX model and DXNN
### 3. Added
- chore: support '--debug' flag in 'dxmz eval' to enable detailed logs

## v0.1.5 / 2025-06-04
### 1. Changed
- None
### 2. Fixed
- Fix error for Image Denoising Add 'bsd68' EvaluationType in enums.py
### 3. Added
- Add internal scripts (log parser, run eval onnx and so on)

## v0.1.3 / 2025-05-26
### 1. Changed
- support check and install 'jq' and ' curl' in download_onnx_json_from*.sh
- support '--exclude_from', '--include_from' and '--get_model_list' options
### 2. Fixed
- fix problem for SSL error using curl on intranet
### 3. Added
- None

## v0.1.2 / 2025-05-15
### 1. Changed
- None
### 2. Fixed
- fix: Support running setup.sh when using modelzoo standalone (without DX-AS all suite)
- fix: Check Python venv setup and activation status in install.sh and handle errors accordingly
### 3. Added
- None

## [0.1.1] - 2025-05-14

### Fixes

- dxmz eval args validation updated

### Miscellaneous Tasks

- add download dataset & model & auto compile scripts
- update internal excluded
- update internal scripts
- add evaluation scripts
- add internal shell scripts

## [0.1.0] - 2025-05-12

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
