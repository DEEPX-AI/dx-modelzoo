# Datasets

API reference for registered dataset classes in DX-ModelZoo, including
constructors, return formats, and resources for supported datasets.

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

See [DataLoader](04_04_DataLoader_API.md) for `DatasetBase` interface details.

## Summary

The following table lists all registered dataset classes:

| Registry Key | Class | Task | Module |
|-------------|-------|------|--------|
| `AFLW20003D` | `AFLW20003D` | 3D Face Landmark | `dx_modelzoo.dataset.aflw20003d` |
| `ADE20K` | `ADE20K` | Semantic Segmentation | `dx_modelzoo.dataset.ade20k` |
| `BDD100K` | `BDD100K` | Panoptic Driving Perception / Vehicle Detection | `dx_modelzoo.dataset.bdd100k` |
| `BSD100` | `BSD100` | Super Resolution | `dx_modelzoo.dataset.bsd` |
| `BSD68` | `BSD68` | Image Denoising | `dx_modelzoo.dataset.bsd` |
| `CBSD68` | `CBSD68` | Image Denoising | `dx_modelzoo.dataset.bsd` |
| `CamVid` | `CamVid` | Semantic Segmentation | `dx_modelzoo.dataset.camvid` |
| `COCO` | `COCO` | Instance Segmentation | `dx_modelzoo.dataset.coco` |
| `COCOPose` | `COCOPose` | Pose Estimation | `dx_modelzoo.dataset.coco` |
| `COCOPoseTopDown` | `COCOPoseTopDown` | Pose Estimation (Top-Down) | `dx_modelzoo.dataset.coco` |
| `COCOPersonSeg` | `COCOPersonSeg` | Person Segmentation | `dx_modelzoo.dataset.coco` |
| `CelebA` | `CelebA` | Face Attribute | `dx_modelzoo.dataset.celeba` |
| `Cityscapes` | `Cityscapes` | Semantic Segmentation | `dx_modelzoo.dataset.cityscapes` |
| `DOTAv1` | `DOTAv1` | Oriented Object Detection | `dx_modelzoo.dataset.dotav1` |
| `Distinctions646` | `Distinctions646` | Image Matting | `dx_modelzoo.dataset.distinctions646` |
| `HandKeypoints` | `HandKeypoints` | Hand Landmark | `dx_modelzoo.dataset.hand_keypoints` |
| `HandKeypointsDetection` | `HandKeypointsDetection` | Hand Detection | `dx_modelzoo.dataset.hand_keypoints` |
| `HOPE` | `HOPE` | Object Pose Estimation | `dx_modelzoo.dataset.hope` |
| `HPatches` | `HPatches` | Keypoint Detection | `dx_modelzoo.dataset.hpatches` |
| `ILSVRC2012` | `ILSVRC2012` | Image Classification | `dx_modelzoo.dataset.imagenet` |
| `ImageNetText` | `ImageNetText` | Image Retrieval | `dx_modelzoo.dataset.imagenet` |
| `KITTI` | `KITTI` | 3D Object Detection | `dx_modelzoo.dataset.kitti` |
| `LFW` | `LFW` | Face Recognition | `dx_modelzoo.dataset.lfw` |
| `LOL` | `LOL` | Low-Light Enhancement | `dx_modelzoo.dataset.lol` |
| `Market1501` | `Market1501` | Person Re-ID | `dx_modelzoo.dataset.market1501` |
| `MVTecAD` | `MVTecADDataset` | Anomaly Detection | `dx_modelzoo.dataset.mvtec` |
| `NYUDepthv2` | `NYUDepthv2` | Depth Estimation | `dx_modelzoo.dataset.nyu` |
| `Objectron` | `Objectron` | 3D Object Detection / Pose | `dx_modelzoo.dataset.objectron` |
| `OxfordIIITPet` | `OxfordIIITPet` | Semantic Segmentation | `dx_modelzoo.dataset.oxford_iiit_pet` |
| `PETA` | `PETA` | Pedestrian Attribute | `dx_modelzoo.dataset.peta` |
| `PascalVOC2007` | `PascalVOC2007` | Object Detection | `dx_modelzoo.dataset.voc` |
| `PascalVOC2012` | `PascalVOC2012` | Semantic Segmentation | `dx_modelzoo.dataset.voc` |
| `WiderFace` | `WiderFace` | Face Detection | `dx_modelzoo.dataset.widerface` |

## Classification

### ImageNet ILSVRC2012

**Module:** `dx_modelzoo.dataset.imagenet`

**Registry key:** `ILSVRC2012`

**Task:** Image Classification

#### Constructor

```python
class ILSVRC2012(DatasetBase):
    def __init__(self, data_dir: str)
```

| Parameter | Type | Default |
|-----------|------|---------|
| `data_dir` | `str` | (required) |

#### `__getitem__` Return Format

| Index | Element |
|-------|---------|
| `[0]` | `img` |
| `[1]` | `label` |

#### Resources

- **Download**: [ImageNet ILSVRC2012](https://image-net.org/download-images.php)
- **License**: Academic/Research use only

### ImageNet Text Prompts

**Module:** `dx_modelzoo.dataset.imagenet`

**Registry key:** `ImageNetText`

**Task:** Image Retrieval

**Notes:** Converts ImageNet class prompts into CLIP text-transformer inputs
used as text queries against a precomputed image-embedding gallery.

## Object Detection

### DOTA v1

**Module:** `dx_modelzoo.dataset.dotav1`

**Registry key:** `DOTAv1`

**Task:** Oriented Object Detection

#### Constructor

```python
class DOTAv1(DatasetBase):
    def __init__(self, data_dir: str, split: str = 'val')
```

| Parameter | Type | Default |
|-----------|------|---------|
| `data_dir` | `str` | (required) |
| `split` | `str` | `'val'` |

#### `__getitem__` Return Format

| Index | Element |
|-------|---------|
| `[0]` | `img` |
| `[1]` | `origin_img.shape` |
| `[2]` | `self.ids[idx]` |

#### Resources

- **Download**: [DOTA v1](https://captain-whu.github.io/DOTA/dataset.html)
- **License**: Research use only

### PascalVOC2007

**Module:** `dx_modelzoo.dataset.voc`

**Registry key:** `PascalVOC2007`

**Task:** Object Detection

#### Constructor

```python
class PascalVOC2007(DatasetBase):
    def __init__(self, data_dir: str)
```

| Parameter | Type | Default |
|-----------|------|---------|
| `data_dir` | `str` | (required) |

#### `__getitem__` Return Format

| Index | Element |
|-------|---------|
| `[0]` | `self.preprocessing(image)` |
| `[1]` | `image.shape` |
| `[2]` | `idx` |

#### Dependencies

- `faster_coco_eval`

#### Resources

- **Download**: [PascalVOC2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/)
- **License**: "VOC2007" challenge data

### BDD100K

**Module:** `dx_modelzoo.dataset.bdd100k`

**Registry key:** `BDD100K`

**Task:** Panoptic Driving Perception / Vehicle Detection

**Notes:** Provides aligned validation samples for YOLOPv2-style evaluation:
vehicle boxes plus drivable-area and lane masks.

### KITTI

**Module:** `dx_modelzoo.dataset.kitti`

**Registry key:** `KITTI`

**Task:** 3D Object Detection

**Notes:** Converts LiDAR point clouds into a Bird's-Eye-View (BEV) map for
SFA3D-style evaluation. Ground-truth 3D boxes are read from `label_2` and
projected into LiDAR coordinates.

#### Constructor

```python
class KITTI(DatasetBase):
    def __init__(self, data_dir: str, split: str = "val")
```

| Parameter | Type | Default |
|-----------|------|---------|
| `data_dir` | `str` | (required) |
| `split` | `str` | `"val"` |

#### `__getitem__` Return Format

| Index | Element |
|-------|---------|
| `[0]` | `self.preprocessing(bev)` |
| `[1]` | `bev.shape` |
| `[2]` | `sample_id` |

#### Resources

- **Download**: [KITTI 3D Object Detection](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)
- **License**: Research use only

## Segmentation

### COCO 2017

**Module:** `dx_modelzoo.dataset.coco`

**Registry key:** `COCO`

**Task:** Instance Segmentation

#### Constructor

```python
class COCO(DatasetBase):
    def __init__(self, data_dir: str)
```

| Parameter | Type | Default |
|-----------|------|---------|
| `data_dir` | `str` | (required) |

#### `__getitem__` Return Format

| Index | Element |
|-------|---------|
| `[0]` | `img` |
| `[1]` | `origin_shape` |
| `[2]` | `int(self.ids[idx])` |

#### Dependencies

- `faster_coco_eval`

#### Resources

- **Download**: [COCO 2017](https://cocodataset.org/#download)
- **License**: Annotations under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); images remain under their original owners' rights and Flickr Terms of Use. See [COCO Terms of Use](https://cocodataset.org/#termsofuse).

### Cityscapes

**Module:** `dx_modelzoo.dataset.cityscapes`

**Registry key:** `Cityscapes`

**Task:** Semantic Segmentation

#### Constructor

```python
class Cityscapes(DatasetBase):
    def __init__(self, data_dir: str)
```

| Parameter | Type | Default |
|-----------|------|---------|
| `data_dir` | `str` | (required) |

#### `__getitem__` Return Format

| Index | Element |
|-------|---------|
| `[0]` | `img` |
| `[1]` | `label` |

#### Class Constants

- `num_class = 19`

#### Resources

- **Download**: [Cityscapes](https://www.cityscapes-dataset.com/register/)
- **License**: Academic/Research use only

### ADE20K

**Module:** `dx_modelzoo.dataset.ade20k`

**Registry key:** `ADE20K`

**Task:** Semantic Segmentation

**Notes:** SceneParse150 / ADEChallengeData2016 dataset with `num_class = 150`.
Validation labels are remapped from dataset IDs to model indices.

### Oxford-IIIT Pet

**Module:** `dx_modelzoo.dataset.oxford_iiit_pet`

**Registry key:** `OxfordIIITPet`

**Task:** Semantic Segmentation

#### Constructor

```python
class OxfordIIITPet(DatasetBase):
    def __init__(self, data_dir: str)
```

| Parameter | Type | Default |
|-----------|------|---------|
| `data_dir` | `str` | (required) |

#### `__getitem__` Return Format

| Index | Element |
|-------|---------|
| `[0]` | `img` |
| `[1]` | `label` |

#### Class Constants

- `num_class = 3`

#### Resources

- **Download**: [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- **License**: CC BY-SA 4.0 (free for research and commercial use)

### PascalVOC2012

**Module:** `dx_modelzoo.dataset.voc`

**Registry key:** `PascalVOC2012`

**Task:** Semantic Segmentation

#### Constructor

```python
class PascalVOC2012(DatasetBase):
    def __init__(self, data_dir: str)
```

| Parameter | Type | Default |
|-----------|------|---------|
| `data_dir` | `str` | (required) |

#### `__getitem__` Return Format

| Index | Element |
|-------|---------|
| `[0]` | `img` |
| `[1]` | `label` |

#### Class Constants

- `num_class = 21`

#### Dependencies

- `faster_coco_eval`

#### Resources

- **Download**: [PascalVOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/)
- **License**: "VOC2007" challenge data

### CamVid

**Module:** `dx_modelzoo.dataset.camvid`

**Registry key:** `CamVid`

**Task:** Semantic Segmentation

#### Constructor

```python
class CamVid(DatasetBase):
    def __init__(self, data_dir: str, split: str = "val")
```

| Parameter | Type | Default |
|-----------|------|---------|
| `data_dir` | `str` | (required) |
| `split` | `str` | `"val"` |

#### `__getitem__` Return Format

| Index | Element |
|-------|---------|
| `[0]` | `self.preprocessing(image)` |
| `[1]` | `label` |

#### Class Constants

- `num_class = 11`

#### Resources

- **Download**: [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)
- **License**: Research/non-commercial use only

## Pose / Landmark

### AFLW2000-3D

**Module:** `dx_modelzoo.dataset.aflw20003d`

**Registry key:** `AFLW20003D`

**Task:** 3D Face Landmark

#### Constructor

```python
class AFLW20003D(DatasetBase):
    def __init__(self, data_dir: str)
```

| Parameter | Type | Default |
|-----------|------|---------|
| `data_dir` | `str` | (required) |

#### `__getitem__` Return Format

| Index | Element |
|-------|---------|
| `[0]` | `face_crop` |
| `[1]` | `gt_landmarks` |
| `[2]` | `bbox_size` |
| `[3]` | `yaw_deg` |
| `[4]` | `idx` |

#### Class Constants

- `NUM_KEYPOINTS = 68`

#### Dependencies

- `scipy`

#### Resources

- **Download**: [AFLW2000-3D](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm)
- **License**: Research use only

### COCOPose

**Module:** `dx_modelzoo.dataset.coco`

**Registry key:** `COCOPose`

**Task:** Pose Estimation

#### Constructor

```python
class COCOPose(DatasetBase):
    def __init__(self, data_dir: str)
```

| Parameter | Type | Default |
|-----------|------|---------|
| `data_dir` | `str` | (required) |

#### `__getitem__` Return Format

| Index | Element |
|-------|---------|
| `[0]` | `img` |
| `[1]` | `origin_shape` |
| `[2]` | `int(self.ids[idx])` |

#### Dependencies

- `faster_coco_eval`

#### Resources

- **Download**: [COCOPose](https://cocodataset.org/#download)
- **License**: Annotations under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); images remain under their original owners' rights and Flickr Terms of Use. See [COCO Terms of Use](https://cocodataset.org/#termsofuse).

### COCOPoseTopDown

**Module:** `dx_modelzoo.dataset.coco`

**Registry key:** `COCOPoseTopDown`

**Task:** Pose Estimation (Top-Down)

**Notes:** Each sample is a single-person crop from a COCO validation image,
using GT bounding boxes with 25% padding. Keypoints are transformed into crop
coordinates.

#### Constructor

```python
class COCOPoseTopDown(DatasetBase):
    def __init__(
        self,
        data_dir: str,
        bbox_pad_ratio: float = 0.25,
        square_crop: bool = False,
        model_input_w: int = 192,
        model_input_h: int = 256,
    )
```

| Parameter | Type | Default |
|-----------|------|---------|
| `data_dir` | `str` | (required) |
| `bbox_pad_ratio` | `float` | `0.25` |
| `square_crop` | `bool` | `False` |
| `model_input_w` | `int` | `192` |
| `model_input_h` | `int` | `256` |

#### `__getitem__` Return Format

| Index | Element |
|-------|---------|
| `[0]` | `crop` (preprocessed) |
| `[1]` | `gt_bbox` |
| `[2]` | `crop_params` |
| `[3]` | `kpts_crop` |
| `[4]` | `visibility` |
| `[5]` | `img_id` |
| `[6]` | `ann_id` |

#### Dependencies

- `faster_coco_eval`

#### Resources

- **Download**: [COCO 2017](https://cocodataset.org/#download)
- **License**: Annotations under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); images remain under their original owners' rights and Flickr Terms of Use. See [COCO Terms of Use](https://cocodataset.org/#termsofuse).

### HOPE

**Module:** `dx_modelzoo.dataset.hope`

**Registry key:** `HOPE`

**Task:** Object Pose Estimation

**Notes:** HOPE-Image validation dataset for single-object 6-DoF pose
evaluation. Supports DOPE-style projected cuboid keypoints and object geometry.

### Objectron

**Module:** `dx_modelzoo.dataset.objectron`

**Registry key:** `Objectron`

**Task:** 3D Object Detection / Object Pose Estimation

**Notes:** Reads native TFRecord test shards directly and returns 9-point 2D
keypoints for cuboid-based evaluation.

### HPatches

**Module:** `dx_modelzoo.dataset.hpatches`

**Registry key:** `HPatches`

**Task:** Keypoint Detection

**Notes:** Sequence dataset with reference/target image pairs and homographies for
repeatability and matching evaluation.

### Hand Keypoints

**Module:** `dx_modelzoo.dataset.hand_keypoints`

**Registry key:** `HandKeypoints`

**Task:** Hand Landmark

#### Constructor

```python
class HandKeypoints(DatasetBase):
    def __init__(self, data_dir: str)
```

| Parameter | Type | Default |
|-----------|------|---------|
| `data_dir` | `str` | (required) |

#### `__getitem__` Return Format

| Index | Element |
|-------|---------|
| `[0]` | `crop` |
| `[1]` | `gt_kpts.astype(np.float32)` |
| `[2]` | `idx` |

#### Class Constants

- `NUM_KEYPOINTS = 21`

#### Resources

- **License**: Custom dataset

### Hand Keypoints Detection

**Module:** `dx_modelzoo.dataset.hand_keypoints`

**Registry key:** `HandKeypointsDetection`

**Task:** Hand Detection

**Notes:** Whole-image hand detection view of the hand-keypoints dataset —
returns the full preprocessed image with all hand bounding boxes for AP
evaluation (unlike `HandKeypoints`, which crops a single rotated hand).

#### Constructor

```python
class HandKeypointsDetection(DatasetBase):
    def __init__(self, data_dir: str)
```

| Parameter | Type | Default |
|-----------|------|---------|
| `data_dir` | `str` | (required) |

#### `__getitem__` Return Format

| Index | Element |
|-------|---------|
| `[0]` | `image` (preprocessed) |
| `[1]` | `boxes` (normalized `xyxy`) |
| `[2]` | `(h, w)` |
| `[3]` | `img_path` |

#### Resources

- **License**: Custom dataset

## Face

### CelebA

**Module:** `dx_modelzoo.dataset.celeba`

**Registry key:** `CelebA`

**Task:** Face Attribute

#### Constructor

```python
class CelebA(DatasetBase):
    def __init__(self, data_dir: str)
```

| Parameter | Type | Default |
|-----------|------|---------|
| `data_dir` | `str` | (required) |

#### `__getitem__` Return Format

| Index | Element |
|-------|---------|
| `[0]` | `img` |
| `[1]` | `labels` |
| `[2]` | `idx` |

#### Class Constants

- `NUM_ATTRIBUTES = 40`

#### Resources

- **Download**: [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- **License**: Non-commercial research only

### LFW (Labeled Faces in the Wild)

**Module:** `dx_modelzoo.dataset.lfw`

**Registry key:** `LFW`

**Task:** Face Recognition

#### Constructor

```python
class LFW(DatasetBase):
    def __init__(self, data_dir: str)
```

| Parameter | Type | Default |
|-----------|------|---------|
| `data_dir` | `str` | (required) |

#### `__getitem__` Return Format

| Index | Element |
|-------|---------|
| `[0]` | `img1` |
| `[1]` | `img2` |
| `[2]` | `label` |

#### Resources

- **Download**: [LFW (Labeled Faces in the Wild)](http://vis-www.cs.umass.edu/lfw/)
- **License**: Research use only

### WiderFace

**Module:** `dx_modelzoo.dataset.widerface`

**Registry key:** `WiderFace`

**Task:** Face Detection

#### Constructor

```python
class WiderFace(DatasetBase):
    def __init__(self, data_dir: str)
```

| Parameter | Type | Default |
|-----------|------|---------|
| `data_dir` | `str` | (required) |

#### `__getitem__` Return Format

| Index | Element |
|-------|---------|
| `[0]` | `self.preprocessing(img)` |
| `[1]` | `img.shape` |
| `[2]` | `file_path` |

#### Dependencies

- `scipy`

#### Resources

- **Download**: [WiderFace](http://shuoyang1213.me/WIDERFACE/)
- **License**: Non-commercial research only

## Person

### PETA (PEdesTrian Attribute)

**Module:** `dx_modelzoo.dataset.peta`

**Registry key:** `PETA`

**Task:** Pedestrian Attribute

#### Constructor

```python
class PETA(DatasetBase):
    def __init__(self, data_dir: str)
```

| Parameter | Type | Default |
|-----------|------|---------|
| `data_dir` | `str` | (required) |

#### `__getitem__` Return Format

| Index | Element |
|-------|---------|
| `[0]` | `img` |
| `[1]` | `labels` |
| `[2]` | `idx` |

#### Dependencies

- `scipy`

#### Resources

- **Download**: [PETA (PEdesTrian Attribute)](https://mmlab.ie.cuhk.edu.hk/projects/PETA.html)
- **License**: Research use only

### COCOPersonSeg

**Module:** `dx_modelzoo.dataset.coco`

**Registry key:** `COCOPersonSeg`

**Task:** Person Segmentation

**Notes:** Builds a binary person-vs-background mask (0=bg, 1=person) by
rasterizing the union of all COCO `person` instance segmentations.

#### Constructor

```python
class COCOPersonSeg(DatasetBase):
    def __init__(
        self,
        data_dir: str,
        inputs=None,
        ann_file: str | None = None,
        person_only: bool = True,
        **kwargs,
    )
```

| Parameter | Type | Default |
|-----------|------|---------|
| `data_dir` | `str` | (required) |
| `inputs` | `list[dict] \| None` | `None` |
| `ann_file` | `str \| None` | `None` |
| `person_only` | `bool` | `True` |

#### `__getitem__` Return Format

| Index | Element |
|-------|---------|
| `[0]` | `image` (preprocessed) |
| `[1]` | `label` (binary mask) |

#### Class Constants

- `num_class = 2`

#### Dependencies

- `faster_coco_eval`

#### Resources

- **Download**: [COCO 2017](https://cocodataset.org/#download)
- **License**: Annotations under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); images remain under their original owners' rights and Flickr Terms of Use. See [COCO Terms of Use](https://cocodataset.org/#termsofuse).

## Image Restoration

### BSD100

**Module:** `dx_modelzoo.dataset.bsd`

**Registry key:** `BSD100`

**Task:** Super Resolution

#### Constructor

```python
class BSD100(DatasetBase):
    def __init__(self, data_dir: str)
```

| Parameter | Type | Default |
|-----------|------|---------|
| `data_dir` | `str` | (required) |

#### `__getitem__` Return Format

| Index | Element |
|-------|---------|
| `[0]` | `lr_image` |
| `[1]` | `hr_image` |

#### Resources

- **Download**: [BSD100](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)
- **License**: BSD license (free for research and commercial use)

### BSD68

**Module:** `dx_modelzoo.dataset.bsd`

**Registry key:** `BSD68`

**Task:** Image Denoising

#### Constructor

```python
class BSD68(DatasetBase):
    def __init__(self, data_dir: str)
```

| Parameter | Type | Default |
|-----------|------|---------|
| `data_dir` | `str` | (required) |

#### `__getitem__` Return Format

| Index | Element |
|-------|---------|
| `[0]` | `self.preprocessing(origin_img)` |
| `[1]` | `origin_img` |

#### Resources

- **Download**: [BSD68](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)
- **License**: BSD license (free for research and commercial use)

### CBSD68

**Module:** `dx_modelzoo.dataset.bsd`

**Registry key:** `CBSD68`

**Task:** Image Denoising

#### Constructor

```python
class CBSD68(DatasetBase):
    def __init__(self, data_dir: str)
```

| Parameter | Type | Default |
|-----------|------|---------|
| `data_dir` | `str` | (required) |

#### `__getitem__` Return Format

| Index | Element |
|-------|---------|
| `[0]` | `self.preprocessing(origin_img)` |
| `[1]` | `origin_img` |

#### Resources

- **Download**: [CBSD68](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)
- **License**: BSD license (free for research and commercial use)

### LOL (Low-Light)

**Module:** `dx_modelzoo.dataset.lol`

**Registry key:** `LOL`

**Task:** Low-Light Enhancement

#### Constructor

```python
class LOL(DatasetBase):
    def __init__(self, data_dir: str)
```

| Parameter | Type | Default |
|-----------|------|---------|
| `data_dir` | `str` | (required) |

#### `__getitem__` Return Format

| Index | Element |
|-------|---------|
| `[0]` | `low_img` |
| `[1]` | `high_img` |

#### Resources

- **Download**: [LOL (Low-Light)](https://daooshee.github.io/BMVC2018website/)
- **License**: Research use only

## Depth

### NYU Depth v2

**Module:** `dx_modelzoo.dataset.nyu`

**Registry key:** `NYUDepthv2`

**Task:** Depth Estimation

#### Constructor

```python
class NYUDepthv2(DatasetBase):
    def __init__(self, data_dir: str)
```

| Parameter | Type | Default |
|-----------|------|---------|
| `data_dir` | `str` | (required) |

#### `__getitem__` Return Format

| Index | Element |
|-------|---------|
| `[0]` | `self.preprocessing(rgb)` |
| `[1]` | `self.label_preprocessing(depth)` |

#### Dependencies

- `h5py`

#### Resources

- **Download**: [NYU Depth v2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
- **License**: Research use only

## Other

### Market-1501

**Module:** `dx_modelzoo.dataset.market1501`

**Registry key:** `Market1501`

**Task:** Other

#### Constructor

```python
class Market1501(DatasetBase):
    def __init__(self, data_dir: str)
```

| Parameter | Type | Default |
|-----------|------|---------|
| `data_dir` | `str` | (required) |

#### `__getitem__` Return Format

| Index | Element |
|-------|---------|
| `[0]` | `img` |
| `[1]` | `idx` |

#### Resources

- **Download**: [Market-1501](https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html)
- **License**: Research use only

### Distinctions-646

**Module:** `dx_modelzoo.dataset.distinctions646`

**Registry key:** `Distinctions646`

**Task:** Image Matting

**Notes:** Composites foreground/alpha pairs onto PASCAL VOC 2012 backgrounds
in memory; no derived `val/` directory is required.

#### Constructor

```python
class Distinctions646(DatasetBase):
    def __init__(
        self,
        data_dir: str,
        split: str = "val",
        background_dir: str | None = None,
    )
```

| Parameter | Type | Default |
|-----------|------|---------|
| `data_dir` | `str` | (required) |
| `split` | `str` | `"val"` |
| `background_dir` | `str \| None` | `None` (resolves to the sibling PascalVOC dataset under `data_dir`) |

#### `__getitem__` Return Format

| Index | Element |
|-------|---------|
| `[0]` | `self.preprocessing(composite)` |
| `[1]` | `alpha` (float32, normalized `[0, 1]`) |

#### Resources

- **Download**: [Distinctions-646](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.8/Matting) (PaddleSeg PP-Matting) + [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) (backgrounds)
- **License**: Follow Distinctions-646 and PASCAL VOC 2012 terms

### MVTec AD

**Module:** `dx_modelzoo.dataset.mvtec`

**Registry key:** `MVTecAD`

**Task:** Anomaly Detection

**Notes:** Yields one test image per index with its image-level label
(0=good, 1=anomalous), category name, and ground-truth mask path. Also
exposes `train_files()` for building a PatchCore memory bank / EfficientAD
statistics.

#### Constructor

```python
class MVTecADDataset(DatasetBase):
    def __init__(
        self,
        data_dir: str,
        category: Optional[Union[str, List[str]]] = None,
        max_train_images: int = 0,
    )
```

| Parameter | Type | Default |
|-----------|------|---------|
| `data_dir` | `str` | (required) |
| `category` | `str \| list[str] \| None` | `None` (all categories) |
| `max_train_images` | `int` | `0` (use all) |

#### `__getitem__` Return Format

| Index | Element |
|-------|---------|
| `[0]` | `img` (preprocessed) |
| `[1]` | `label` (0=good, 1=anomalous) |
| `[2]` | `category` |
| `[3]` | `mask_path` |

#### Resources

- **Download**: [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- **License**: Non-commercial research/education use only
