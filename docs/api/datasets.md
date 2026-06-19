# Datasets

API reference for all dataset classes in DX-ModelZoo.

**Module:** `dx_modelzoo.dataset`

## Overview

All datasets inherit from `DatasetBase` and are registered via `DATASET_REGISTRY`.
Each class is registered with `@DATASET_REGISTRY.register` — if no explicit key is
provided, the class name is used as the registry key.

```python
from dx_modelzoo.dataset import DATASET_REGISTRY

# Look up a dataset class by registry key
dataset_cls = DATASET_REGISTRY.get("ILSVRC2012")
dataset = dataset_cls(data_dir="/path/to/data")
```

See [DataLoader](dataloader.md) for `DatasetBase` interface details.

## Summary

| Registry Key | Class | Task | Module |
|-------------|-------|------|--------|
| `AFLW20003D` | `AFLW20003D` | 3D Face Landmark | `dx_modelzoo.dataset.aflw20003d` |
| `ADE20K` | `ADE20K` | Semantic Segmentation | `dx_modelzoo.dataset.ade20k` |
| `BDD100K` | `BDD100K` | Panoptic Driving Perception / Vehicle Detection | `dx_modelzoo.dataset.bdd100k` |
| `BSD100` | `BSD100` | Super Resolution | `dx_modelzoo.dataset.bsd` |
| `BSD68` | `BSD68` | Image Denoising | `dx_modelzoo.dataset.bsd` |
| `CBSD68` | `CBSD68` | Image Denoising | `dx_modelzoo.dataset.bsd` |
| `COCO` | `COCO` | Instance Segmentation | `dx_modelzoo.dataset.coco` |
| `COCOMultiInput` | `COCOMultiInput` | Multi-input Detection | `dx_modelzoo.dataset.coco_multiinput` |
| `COCOPose` | `COCOPose` | Pose Estimation | `dx_modelzoo.dataset.coco` |
| `COCOPoseTopDown` | `COCOPoseTopDown` | Pose Estimation (Top-Down) | `dx_modelzoo.dataset.coco` |
| `COCOPersonSeg` | `COCOPersonSeg` | Person Segmentation | `dx_modelzoo.dataset.coco` |
| `CelebA` | `CelebA` | Face Attribute | `dx_modelzoo.dataset.celeba` |
| `Cityscapes` | `Cityscapes` | Semantic Segmentation | `dx_modelzoo.dataset.cityscapes` |
| `DOTAv1` | `DOTAv1` | Oriented Object Detection | `dx_modelzoo.dataset.dotav1` |
| `HandKeypoints` | `HandKeypoints` | Hand Landmark | `dx_modelzoo.dataset.hand_keypoints` |
| `HandKeypointsDetection` | `HandKeypointsDetection` | Hand Detection | `dx_modelzoo.dataset.hand_keypoints` |
| `HOPE` | `HOPE` | Object Pose Estimation | `dx_modelzoo.dataset.hope` |
| `HPatches` | `HPatches` | Keypoint Detection | `dx_modelzoo.dataset.hpatches` |
| `ILSVRC2012` | `ILSVRC2012` | Image Classification | `dx_modelzoo.dataset.imagenet` |
| `LFW` | `LFW` | Face Recognition | `dx_modelzoo.dataset.lfw` |
| `LOL` | `LOL` | Low-Light Enhancement | `dx_modelzoo.dataset.lol` |
| `Market1501` | `Market1501` | Other | `dx_modelzoo.dataset.market1501` |
| `NYUDepthv2` | `NYUDepthv2` | Depth Estimation | `dx_modelzoo.dataset.nyu` |
| `Objectron` | `Objectron` | 3D Object Detection / Pose | `dx_modelzoo.dataset.objectron` |
| `OxfordIIITPet` | `OxfordIIITPet` | Semantic Segmentation | `dx_modelzoo.dataset.oxford_iiit_pet` |
| `PETA` | `PETA` | Pedestrian Attribute | `dx_modelzoo.dataset.peta` |
| `PascalVOC2007` | `PascalVOC2007` | Object Detection | `dx_modelzoo.dataset.voc` |
| `PascalVOC2012` | `PascalVOC2012` | Semantic Segmentation | `dx_modelzoo.dataset.voc` |
| `SyntheticMultiInput` | `SyntheticMultiInput` | Multi-input Synthetic Data | `dx_modelzoo.dataset.synthetic` |
| `WiderFace` | `WiderFace` | Face Detection | `dx_modelzoo.dataset.widerface` |

---

# Classification

## ImageNet ILSVRC2012

**Module:** `dx_modelzoo.dataset.imagenet`

**Registry key:** `ILSVRC2012`

**Task:** Image Classification

### Constructor

```python
class ILSVRC2012(DatasetBase):
    def __init__(self, data_dir: str)
```

| Parameter | Type | Default |
|-----------|------|---------|
| `data_dir` | `str` | (required) |

### `__getitem__` Return Format

| Index | Element |
|-------|---------|
| `[0]` | `img` |
| `[1]` | `label` |

### Resources

- **Download**: [ImageNet ILSVRC2012](https://image-net.org/download-images.php)
- **License**: Academic/Research use only

---

---

# Object Detection

## DOTA v1

**Module:** `dx_modelzoo.dataset.dotav1`

**Registry key:** `DOTAv1`

**Task:** Oriented Object Detection

### Constructor

```python
class DOTAv1(DatasetBase):
    def __init__(self, data_dir: str, split: str = 'val')
```

| Parameter | Type | Default |
|-----------|------|---------|
| `data_dir` | `str` | (required) |
| `split` | `str` | `'val'` |

### `__getitem__` Return Format

| Index | Element |
|-------|---------|
| `[0]` | `img` |
| `[1]` | `origin_img.shape` |
| `[2]` | `self.ids[idx]` |

### Resources

- **Download**: [DOTA v1](https://captain-whu.github.io/DOTA/dataset.html)
- **License**: Research use only

---

## PascalVOC2007

**Module:** `dx_modelzoo.dataset.voc`

**Registry key:** `PascalVOC2007`

**Task:** Object Detection

### Constructor

```python
class PascalVOC2007(DatasetBase):
    def __init__(self, data_dir: str)
```

| Parameter | Type | Default |
|-----------|------|---------|
| `data_dir` | `str` | (required) |

### `__getitem__` Return Format

| Index | Element |
|-------|---------|
| `[0]` | `self.preprocessing(image)` |
| `[1]` | `image.shape` |
| `[2]` | `idx` |

### Dependencies

- `faster_coco_eval (optional)`

### Resources

- **Download**: [PascalVOC2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/)
- **License**: "VOC2007" challenge data

---

## BDD100K

**Module:** `dx_modelzoo.dataset.bdd100k`

**Registry key:** `BDD100K`

**Task:** Panoptic Driving Perception / Vehicle Detection

**Notes:** Provides aligned validation samples for YOLOPv2-style evaluation:
vehicle boxes plus drivable-area and lane masks.

---

## COCOMultiInput

**Module:** `dx_modelzoo.dataset.coco_multiinput`

**Registry key:** `COCOMultiInput`

**Task:** Multi-input Detection

**Notes:** COCO val2017 wrapper that returns a dict of named tensors for models
with multiple inputs or control scalars.

---

---

# Segmentation

## COCO 2017

**Module:** `dx_modelzoo.dataset.coco`

**Registry key:** `COCO`

**Task:** Instance Segmentation

### Constructor

```python
class COCO(DatasetBase):
    def __init__(self, data_dir: str)
```

| Parameter | Type | Default |
|-----------|------|---------|
| `data_dir` | `str` | (required) |

### `__getitem__` Return Format

| Index | Element |
|-------|---------|
| `[0]` | `img` |
| `[1]` | `origin_shape` |
| `[2]` | `int(self.ids[idx]` |

### Dependencies

- `faster_coco_eval (optional)`

### Resources

- **Download**: [COCO 2017](https://cocodataset.org/#download)
- **License**: CC BY 4.0 (free for research and commercial use)

---

## Cityscapes

**Module:** `dx_modelzoo.dataset.cityscapes`

**Registry key:** `Cityscapes`

**Task:** Semantic Segmentation

### Constructor

```python
class Cityscapes(DatasetBase):
    def __init__(self, data_dir: str)
```

| Parameter | Type | Default |
|-----------|------|---------|
| `data_dir` | `str` | (required) |

### `__getitem__` Return Format

| Index | Element |
|-------|---------|
| `[0]` | `img` |
| `[1]` | `label` |

### Class Constants

- `num_class = 19`

### Resources

- **Download**: [Cityscapes](https://www.cityscapes-dataset.com/register/)
- **License**: Academic/Research use only

---

## ADE20K

**Module:** `dx_modelzoo.dataset.ade20k`

**Registry key:** `ADE20K`

**Task:** Semantic Segmentation

**Notes:** SceneParse150 / ADEChallengeData2016 dataset with `num_class = 150`.
Validation labels are remapped from dataset IDs to model indices.

---

## Oxford-IIIT Pet

**Module:** `dx_modelzoo.dataset.oxford_iiit_pet`

**Registry key:** `OxfordIIITPet`

**Task:** Semantic Segmentation

### Constructor

```python
class OxfordIIITPet(DatasetBase):
    def __init__(self, data_dir: str)
```

| Parameter | Type | Default |
|-----------|------|---------|
| `data_dir` | `str` | (required) |

### `__getitem__` Return Format

| Index | Element |
|-------|---------|
| `[0]` | `img` |
| `[1]` | `label` |

### Class Constants

- `num_class = 3`

### Resources

- **Download**: [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- **License**: CC BY-SA 4.0 (free for research and commercial use)

---

## PascalVOC2012

**Module:** `dx_modelzoo.dataset.voc`

**Registry key:** `PascalVOC2012`

**Task:** Semantic Segmentation

### Constructor

```python
class PascalVOC2012(DatasetBase):
    def __init__(self, data_dir: str)
```

| Parameter | Type | Default |
|-----------|------|---------|
| `data_dir` | `str` | (required) |

### `__getitem__` Return Format

| Index | Element |
|-------|---------|
| `[0]` | `img` |
| `[1]` | `label` |

### Class Constants

- `num_class = 21`

### Dependencies

- `faster_coco_eval (optional)`

### Resources

- **Download**: [PascalVOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/)
- **License**: "VOC2007" challenge data

---

---

# Pose / Landmark

## AFLW2000-3D

**Module:** `dx_modelzoo.dataset.aflw20003d`

**Registry key:** `AFLW20003D`

**Task:** 3D Face Landmark

### Constructor

```python
class AFLW20003D(DatasetBase):
    def __init__(self, data_dir: str)
```

| Parameter | Type | Default |
|-----------|------|---------|
| `data_dir` | `str` | (required) |

### `__getitem__` Return Format

| Index | Element |
|-------|---------|
| `[0]` | `face_crop` |
| `[1]` | `gt_landmarks` |
| `[2]` | `bbox_size` |
| `[3]` | `yaw_deg` |
| `[4]` | `idx` |

### Class Constants

- `NUM_KEYPOINTS = 68`

### Dependencies

- `scipy`

### Resources

- **Download**: [AFLW2000-3D](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm)
- **License**: Research use only

---

## COCOPose

**Module:** `dx_modelzoo.dataset.coco`

**Registry key:** `COCOPose`

**Task:** Pose Estimation

### Constructor

```python
class COCOPose(DatasetBase):
    def __init__(self, data_dir: str)
```

| Parameter | Type | Default |
|-----------|------|---------|
| `data_dir` | `str` | (required) |

### `__getitem__` Return Format

| Index | Element |
|-------|---------|
| `[0]` | `img` |
| `[1]` | `origin_shape` |
| `[2]` | `int(self.ids[idx]` |

### Dependencies

- `faster_coco_eval (optional)`

### Resources

- **Download**: [COCOPose](https://cocodataset.org/#download)
- **License**: CC BY 4.0 (free for research and commercial use)

---

## HOPE

**Module:** `dx_modelzoo.dataset.hope`

**Registry key:** `HOPE`

**Task:** Object Pose Estimation

**Notes:** HOPE-Image validation dataset for single-object 6-DoF pose
evaluation. Supports DOPE-style projected cuboid keypoints and object geometry.

---

## HPatches

**Module:** `dx_modelzoo.dataset.hpatches`

**Registry key:** `HPatches`

**Task:** Keypoint Detection

**Notes:** Sequence dataset with reference/target image pairs and homographies for
repeatability and matching evaluation.

---

## Hand Keypoints

**Module:** `dx_modelzoo.dataset.hand_keypoints`

**Registry key:** `HandKeypoints`

**Task:** Hand Landmark

### Constructor

```python
class HandKeypoints(DatasetBase):
    def __init__(self, data_dir: str)
```

| Parameter | Type | Default |
|-----------|------|---------|
| `data_dir` | `str` | (required) |

### `__getitem__` Return Format

| Index | Element |
|-------|---------|
| `[0]` | `crop` |
| `[1]` | `gt_kpts.astype(np.float32)` |
| `[2]` | `idx` |

### Class Constants

- `NUM_KEYPOINTS = 21`

### Resources

- **License**: Custom dataset

---

---

# Face

## CelebA

**Module:** `dx_modelzoo.dataset.celeba`

**Registry key:** `CelebA`

**Task:** Face Attribute

### Constructor

```python
class CelebA(DatasetBase):
    def __init__(self, data_dir: str)
```

| Parameter | Type | Default |
|-----------|------|---------|
| `data_dir` | `str` | (required) |

### `__getitem__` Return Format

| Index | Element |
|-------|---------|
| `[0]` | `img` |
| `[1]` | `labels` |
| `[2]` | `idx` |

### Class Constants

- `NUM_ATTRIBUTES = 40`

### Resources

- **Download**: [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- **License**: Non-commercial research only

---

## LFW (Labeled Faces in the Wild)

**Module:** `dx_modelzoo.dataset.lfw`

**Registry key:** `LFW`

**Task:** Face Recognition

### Constructor

```python
class LFW(DatasetBase):
    def __init__(self, data_dir: str)
```

| Parameter | Type | Default |
|-----------|------|---------|
| `data_dir` | `str` | (required) |

### `__getitem__` Return Format

| Index | Element |
|-------|---------|
| `[0]` | `img1` |
| `[1]` | `img2` |
| `[2]` | `label` |

### Resources

- **Download**: [LFW (Labeled Faces in the Wild)](http://vis-www.cs.umass.edu/lfw/)
- **License**: Research use only

---

## WiderFace

**Module:** `dx_modelzoo.dataset.widerface`

**Registry key:** `WiderFace`

**Task:** Face Detection

### Constructor

```python
class WiderFace(DatasetBase):
    def __init__(self, data_dir: str)
```

| Parameter | Type | Default |
|-----------|------|---------|
| `data_dir` | `str` | (required) |

### `__getitem__` Return Format

| Index | Element |
|-------|---------|
| `[0]` | `self.preprocessing(img)` |
| `[1]` | `img.shape` |
| `[2]` | `file_path` |

### Dependencies

- `scipy`

### Resources

- **Download**: [WiderFace](http://shuoyang1213.me/WIDERFACE/)
- **License**: Non-commercial research only

---

---

# Person

## PETA (PEdesTrian Attribute)

**Module:** `dx_modelzoo.dataset.peta`

**Registry key:** `PETA`

**Task:** Pedestrian Attribute

### Constructor

```python
class PETA(DatasetBase):
    def __init__(self, data_dir: str)
```

| Parameter | Type | Default |
|-----------|------|---------|
| `data_dir` | `str` | (required) |

### `__getitem__` Return Format

| Index | Element |
|-------|---------|
| `[0]` | `img` |
| `[1]` | `labels` |
| `[2]` | `idx` |

### Dependencies

- `scipy`

### Resources

- **Download**: [PETA (PEdesTrian Attribute)](https://mmlab.ie.cuhk.edu.hk/projects/PETA.html)
- **License**: Research use only

---

---

# Image Restoration

## BSD100

**Module:** `dx_modelzoo.dataset.bsd`

**Registry key:** `BSD100`

**Task:** Super Resolution

### Constructor

```python
class BSD100(DatasetBase):
    def __init__(self, data_dir: str)
```

| Parameter | Type | Default |
|-----------|------|---------|
| `data_dir` | `str` | (required) |

### `__getitem__` Return Format

| Index | Element |
|-------|---------|
| `[0]` | `lr_image` |
| `[1]` | `hr_image` |

### Resources

- **Download**: [BSD100](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)
- **License**: BSD license (free for research and commercial use)

---

## BSD68

**Module:** `dx_modelzoo.dataset.bsd`

**Registry key:** `BSD68`

**Task:** Image Denoising

### Constructor

```python
class BSD68(DatasetBase):
    def __init__(self, data_dir: str)
```

| Parameter | Type | Default |
|-----------|------|---------|
| `data_dir` | `str` | (required) |

### `__getitem__` Return Format

| Index | Element |
|-------|---------|
| `[0]` | `self.preprocessing(origin_img)` |
| `[1]` | `origin_img` |

### Resources

- **Download**: [BSD68](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)
- **License**: BSD license (free for research and commercial use)

---

## CBSD68

**Module:** `dx_modelzoo.dataset.bsd`

**Registry key:** `CBSD68`

**Task:** Image Denoising

### Constructor

```python
class CBSD68(DatasetBase):
    def __init__(self, data_dir: str)
```

| Parameter | Type | Default |
|-----------|------|---------|
| `data_dir` | `str` | (required) |

### `__getitem__` Return Format

| Index | Element |
|-------|---------|
| `[0]` | `self.preprocessing(origin_img)` |
| `[1]` | `origin_img` |

### Resources

- **Download**: [CBSD68](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)
- **License**: BSD license (free for research and commercial use)

---

## LOL (Low-Light)

**Module:** `dx_modelzoo.dataset.lol`

**Registry key:** `LOL`

**Task:** Low-Light Enhancement

### Constructor

```python
class LOL(DatasetBase):
    def __init__(self, data_dir: str)
```

| Parameter | Type | Default |
|-----------|------|---------|
| `data_dir` | `str` | (required) |

### `__getitem__` Return Format

| Index | Element |
|-------|---------|
| `[0]` | `low_img` |
| `[1]` | `high_img` |

### Resources

- **Download**: [LOL (Low-Light)](https://daooshee.github.io/BMVC2018website/)
- **License**: Research use only

---

---

# Depth

## NYU Depth v2

**Module:** `dx_modelzoo.dataset.nyu`

**Registry key:** `NYUDepthv2`

**Task:** Depth Estimation

### Constructor

```python
class NYUDepthv2(DatasetBase):
    def __init__(self, data_dir: str)
```

| Parameter | Type | Default |
|-----------|------|---------|
| `data_dir` | `str` | (required) |

### `__getitem__` Return Format

| Index | Element |
|-------|---------|
| `[0]` | `self.preprocessing(rgb)` |
| `[1]` | `self.label_preprocessing(depth` |

### Dependencies

- `h5py`

### Resources

- **Download**: [NYU Depth v2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
- **License**: Research use only

---

## Objectron

**Module:** `dx_modelzoo.dataset.objectron`

**Registry key:** `Objectron`

**Task:** 3D Object Detection / Object Pose Estimation

**Notes:** Reads native TFRecord test shards directly and returns 9-point 2D
keypoints for cuboid-based evaluation.

---

---

# Other

## Market-1501

**Module:** `dx_modelzoo.dataset.market1501`

**Registry key:** `Market1501`

**Task:** Other

### Constructor

```python
class Market1501(DatasetBase):
    def __init__(self, data_dir: str)
```

| Parameter | Type | Default |
|-----------|------|---------|
| `data_dir` | `str` | (required) |

### `__getitem__` Return Format

| Index | Element |
|-------|---------|
| `[0]` | `img` |
| `[1]` | `idx` |

### Resources

- **Download**: [Market-1501](https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html)
- **License**: Research use only

---

## SyntheticMultiInput

**Module:** `dx_modelzoo.dataset.synthetic`

**Registry key:** `SyntheticMultiInput`

**Task:** Multi-input Synthetic Data

**Notes:** Generates reproducible synthetic tensors from the YAML `inputs` spec.
Useful for throughput checks and models without a readily available real dataset.

---
