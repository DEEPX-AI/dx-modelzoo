# Postprocessing

The postprocessing pipeline in DX-ModelZoo transforms raw model outputs into structured, task-specific results.

**Module:** `dx_modelzoo.postprocessing`

!!! note "See Also"
    - [YAML Configuration](../guides/yaml-config.md) - Postprocessing configuration in YAML
    - [Custom Models](../guides/custom-models.md) - Creating custom postprocessing operations
    - [ModelBuilder](model-builder.md) - How to build postprocessing pipelines
    - [Preprocessing](preprocessing.md) - Input data preprocessing (compare NPU skip behavior)

## PostprocessingPipeline

Sequentially executes postprocessing operations defined in a YAML configuration.

```python
class PostprocessingPipeline:
    def __init__(self, steps: list)
    def __call__(self, outputs) -> Any
```

| Attribute | Type | Description |
|-----------|------|-------------|
| `steps` | `list` | List of postprocessing operation instances |

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

**Input:** `list[np.ndarray]` — Model output (logits/probabilities)
**Output:** Top-k indices and values

### `nms`

Applies Non-Maximum Suppression to remove duplicate detections.

```yaml
postprocessing:
  - type: nms
    variant: yolo
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `variant` | `str` | NMS variant (e.g., `yolo`, `ssd`, depending on the model architecture) |

**Task:** Object Detection

### `segmentation_argmax`

Applies argmax along the channel axis to the output of a segmentation model.

```yaml
postprocessing:
  - type: segmentation_argmax
```

No parameters.

**Input:** Logits of shape `(1, num_classes, H, W)`
**Output:** Class index map of shape `(H, W)`

**Task:** Semantic Segmentation

## Postprocessing Guide by Task

| Task | Recommended Postprocessing | YAML Example |
|------|---------------------------|--------------|
| Image Classification | `topk` | `type: topk, k: [1, 5]` |
| Object Detection | model-specific decode + shared `nms` | `type: yolov8_decode` → `type: nms, variant: yolo` |
| Semantic Segmentation | `segmentation_argmax` | `type: segmentation_argmax` |
| Instance Segmentation | model-specific decode | `type: yolact_decode` |
| Face Detection | model-specific decode | `type: retinaface_decode` |
| Pose Estimation | model-specific decode | `type: rtmpose_simcc_decode` |
| Super Resolution | `identity` | `type: identity` |
| Depth Estimation | `identity` | `type: identity` |

## Adding Custom Postprocessing

```python
from dx_modelzoo.postprocessing import POSTPROCESSING_REGISTRY

@POSTPROCESSING_REGISTRY.register("my_postprocess")
class MyPostprocess:
    """Custom postprocessing operation"""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def __call__(self, outputs: list[np.ndarray]) -> dict:
        # outputs: list of raw model outputs
        result = outputs[0]
        mask = result > self.threshold
        return {"predictions": result[mask]}
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
    variant: yolo
```

!!! note "Postprocessing Order"
    Postprocessing operations are executed in the order they are defined in the YAML configuration. The output of each step becomes the input to the next, so ordering matters.

For detailed usage, refer to the [Custom Models](../guides/custom-models.md) guide.

## Model-Specific Postprocessing (`custom_ops.py`)

For model-specific decoding logic that doesn't belong in the shared
postprocessing module, use a `custom_ops.py` file in the model's directory.
Many model YAMLs reference names registered there instead of shared built-ins.

### Auto-Discovery Mechanism

When `ModelBuilder` loads a model YAML, it automatically imports `custom_ops.py` from the same directory:

```
src/dx_modelzoo/models/cv/detection/MyModel/
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
| `db_text_decode` | PP-OCRv5 Det | Probability map → `List[polygon]` |
| `ctc_greedy_decode` | PP-OCRv5 Rec | Logits → `str` |
| `blazeface_decode` | BlazeFace | Raw boxes → `[N, 5]` (xyxy + score) |
| `efficientad_anomaly_map` | EfficientAD | Teacher+Student → `float` score |

### When NOT to Use `POSTPROCESSING_REGISTRY`

Not all model-specific code belongs in the postprocessing registry:

- **Init-time utilities** (run once at startup, not per-sample): Place in a utility file.
  - Example: `clip_utils.py` for building text embeddings at evaluation init time.
- **Common operations** (used across many models): Keep in `src/dx_modelzoo/postprocessing/`.
  - Example: NMS, TopK, coordinate scaling.
