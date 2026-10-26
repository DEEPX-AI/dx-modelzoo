# Postprocessing

The postprocessing pipeline in DX-ModelZoo transforms raw model outputs into structured, task-specific results.

**Module:** `dx_modelzoo.postprocessing`

## PostprocessingPipeline

Sequentially executes postprocessing operations defined in a YAML configuration.

```python
class PostprocessingPipeline:
    def __init__(self, steps_config: list[dict])
    def __call__(self, outputs, **kwargs) -> Any
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `steps_config` | `list[dict]` | Step **configuration** dicts (`{"type": ..., **params}`), i.e. the YAML `postprocessing:` block |

| Attribute | Type | Description |
|-----------|------|-------------|
| `steps` | `list` | Instantiated postprocessing operation objects built from `steps_config` |

!!! warning  
    The constructor argument is `steps_config` and takes **config dicts**, not
    operation instances. Using `steps=` raises `TypeError`; passing built objects
    raises `AttributeError: 'TopK' object has no attribute 'get'`.

Context keyword arguments (`runtime_target`, and for detection models
`origin_hw` / `input_hw`) are forwarded to every step — see
[Adding Custom Postprocessing](#adding-custom-postprocessing).

```python
from dx_modelzoo.postprocessing import PostprocessingPipeline

postprocessing = PostprocessingPipeline(steps_config=[{"type": "topk", "k": 5}])
```

### Usage Example

```python
from dx_modelzoo.loader.model_builder import ModelBuilder

builder = ModelBuilder("resnet50_224x224.yaml")
postprocessing = builder.build_postprocessing()

# Apply to model outputs
outputs = session.run(input_tensor)
result = postprocessing(outputs)
```

!!! note "No NPU Skip"
    Unlike preprocessing, the postprocessing pipeline does not have NPU skip logic. All postprocessing operations are always executed.

## Built-in Postprocessing Types

The shared built-in types below are the only operations registered directly in
`dx_modelzoo.postprocessing.POSTPROCESSING_REGISTRY`: `identity`, `topk`,
`nms`, and `segmentation_argmax`.

### `identity`

Returns the input as-is without any transformation. Used for models that do not require postprocessing.

```yaml
postprocessing:
  - type: identity
```

No parameters.

### `topk`

Extracts the top-k predictions from a classification model.

```yaml
postprocessing:
  - type: topk
    k: [1, 5]
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `k` | `list[int]` | Number of top predictions to extract (e.g., `[1, 5]` → Top-1, Top-5) |
| `skip_background` | `bool` | Drop index 0 (background class) before ranking (default: `False`) |

**Input:** `list[np.ndarray]` — Model output (logits/probabilities)
**Output:** Top-k class indices, sorted by confidence descending

### `nms`

Applies Non-Maximum Suppression to remove duplicate detections.

```yaml
postprocessing:
  - type: nms
    conf_thres: 0.001
    iou_thres: 0.7
    max_output_boxes: 300
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `conf_thres` | `float` | Confidence threshold to keep a detection (default: `0.001`) |
| `iou_thres` | `float` | IoU threshold above which overlapping boxes are suppressed (default: `0.7`) |
| `max_output_boxes` | `int` | Maximum number of boxes to keep per image (default: `300`) |
| `pad_resize` | `bool` | Whether to rescale coordinates for letterbox/pad-resize preprocessing (default: `True`) |

**Input:** `dict` — `{"boxes": [N,4] xyxy, "scores": [N], "class_ids": [N]}` from a model-specific decode step
**Output:** `np.ndarray` of shape `[M, 6]` — `(x1, y1, x2, y2, score, class_id)`

**Task:** Object Detection

### `segmentation_argmax`

Applies argmax along the channel axis to the output of a segmentation model.

```yaml
postprocessing:
  - type: segmentation_argmax
    layout: nchw
    target_size: [512, 512]
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `layout` | `str` | Input layout, `nchw` or `nhwc` (default: `nchw`) |
| `target_size` | `list[int] \| None` | Optional `[H, W]` to resize the output via nearest interpolation (default: `None`) |

**Input:** Logits of shape `(1, num_classes, H, W)`
**Output:** Class index map of shape `(H, W)`

**Task:** Semantic Segmentation

## Postprocessing Guide by Task

The following table maps common tasks to their typical postprocessing:

| Task | Recommended Postprocessing | YAML Example |
|------|---------------------------|--------------|
| Image Classification | `topk` | `type: topk, k: [1, 5]` |
| Object Detection | model-specific decode + shared `nms` | `type: yolov8_decode` → `type: nms, conf_thres: 0.001, iou_thres: 0.7` |
| Semantic Segmentation | `segmentation_argmax` | `type: segmentation_argmax` |
| Instance Segmentation | model-specific decode | `type: yolact_decode` |
| Face Detection | model-specific decode | `type: retinaface_decode` |
| Pose Estimation | model-specific decode | `type: rtmpose_simcc_decode` |
| Super Resolution | `identity` | `type: identity` |
| Depth Estimation | `identity` | `type: identity` |

## Adding Custom Postprocessing

Register a custom postprocessing operation the same way as preprocessing:

```python
from dx_modelzoo.postprocessing import POSTPROCESSING_REGISTRY

@POSTPROCESSING_REGISTRY.register("my_postprocess")
class MyPostprocess:
    """Custom postprocessing operation"""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def __call__(self, outputs: list[np.ndarray], **kwargs) -> dict:
        # outputs: list of raw model outputs
        # kwargs carries pipeline context (see table below) — always accept **kwargs
        result = outputs[0]
        mask = result > self.threshold
        return {"predictions": result[mask]}
```

!!! warning "`__call__` must accept `**kwargs`"  
    `PostprocessingPipeline.__call__(outputs, **kwargs)` forwards its context to
    every step. A step defined as `__call__(self, outputs)` raises
    `TypeError: __call__() got an unexpected keyword argument 'runtime_target'`.

| Keyword | Provided by | Meaning |
|---------|-------------|---------|
| `runtime_target` | all evaluators | `"onnx"` / `"npu"` — the active runtime |
| `origin_hw` | detection-style evaluators | `(H, W)` of the original image |
| `input_hw` | detection-style evaluators | `(H, W)` of the model input |

Steps producing coordinates (boxes, keypoints) must map model-input pixel space
back to original image coordinates using `origin_hw` / `input_hw`:

```python
from dx_modelzoo.postprocessing.coord_scaler import unpad_and_scale

@POSTPROCESSING_REGISTRY.register("my_box_postprocess")
class MyBoxPostprocess:
    def __call__(self, outputs, origin_hw=None, input_hw=None, **kwargs):
        boxes = outputs[0]  # [M, 6] (x1, y1, x2, y2, score, class_id) in input space
        if origin_hw is not None and input_hw is not None:
            boxes[:, :4] = unpad_and_scale(boxes[:, :4], input_hw, origin_hw)
        return boxes
```

```yaml
postprocessing:
  - type: my_postprocess
    threshold: 0.7
```

### Chaining Multiple Postprocessing Steps

Multiple postprocessing operations can be chained sequentially:

```yaml
postprocessing:
  - type: yolov8_decode       # Step 1: model-specific decode
  - type: nms                 # Step 2: shared NMS
    conf_thres: 0.001
    iou_thres: 0.7
```

!!! note "Postprocessing Order"  
    Postprocessing operations are executed in the order they are defined in the YAML configuration. The output of each step becomes the input to the next, so ordering matters.

For detailed usage, refer to the [Custom Models](02_05_Custom_Models.md) guide.

## Model-Specific Postprocessing (`custom_ops.py`)

For model-specific decoding logic that doesn't belong in the shared
postprocessing module, use a `custom_ops.py` file in the model's directory.
Many model YAMLs reference names registered there instead of shared built-ins.

### Auto-Discovery Mechanism

When `ModelBuilder` loads a model YAML, it automatically imports `custom_ops.py` from the same directory:

```
models/custom/MyModel/
├── MyModel_v1.yaml
└── custom_ops.py          ← auto-imported
```

The import happens before validation, so any `@POSTPROCESSING_REGISTRY.register()` calls in `custom_ops.py` are available for the YAML's `postprocessing` section.

### Example: Model-Specific Decoder

```python title="models/cv/pose_estimation/RTMPose/custom_ops.py"
from dx_modelzoo.postprocessing import POSTPROCESSING_REGISTRY
import numpy as np

@POSTPROCESSING_REGISTRY.register("rtmpose_simcc_decode")
class RTMPoseSimCCDecode:
    """Decode SimCC heatmaps to COCO 17-keypoint format."""

    def __init__(self, split_ratio: float = 2.0, **kwargs):
        self.split_ratio = split_ratio

    def __call__(self, outputs, **kwargs):
        simcc_x, simcc_y = outputs[0], outputs[1]
        x_idx = simcc_x[0].argmax(axis=-1).astype(np.float32)
        y_idx = simcc_y[0].argmax(axis=-1).astype(np.float32)
        scores = (simcc_x[0].max(axis=-1) + simcc_y[0].max(axis=-1)) / 2.0
        kpts = np.stack([x_idx / self.split_ratio, y_idx / self.split_ratio], axis=-1)
        return kpts, scores
```

```yaml title="RTMPose_T.yaml (postprocessing section)"
postprocessing:
  - type: rtmpose_simcc_decode
    split_ratio: 2.0
```

### Built-in Model-Specific Types

These types are registered via `custom_ops.py` in their respective model directories:

| Type | Model | Input → Output |
|------|-------|----------------|
| `superpoint_decode` | SuperPoint | 65-ch heatmap → `(keypoints, descriptors)` |
| `rtmpose_simcc_decode` | RTMPose | SimCC bins → `(keypoints [17,2], scores [17])` |
| `mediapipe_pose_decode` | MediaPipePose | Landmarks → `(keypoints [17,2], scores [17])` |
| `blazeface_decode` | BlazeFace | Raw boxes → `[N, 5]` (xyxy + score) |

### When NOT to Use `POSTPROCESSING_REGISTRY`

Not all model-specific code belongs in the postprocessing registry:

- **Init-time utilities** (run once at startup, not per-sample): Place in a utility file.
  - Example: `clip_utils.py` for building text embeddings at evaluation init time.
- **Common operations** (used across many models): Keep in `src/dx_modelzoo/postprocessing/`.
  - Example: NMS, TopK, coordinate scaling.
