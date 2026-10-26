# Datasets

Configuration reference and download instructions for commonly used evaluation
datasets in DX-ModelZoo. For the complete list of all registered datasets, see
the [Datasets API](04_05_Datasets_API.md#summary).

!!! note "Quick Setup"  
    Create symlinks to your actual dataset locations:
    ```bash
    export DATA_ROOT=/your/datasets/folder
    # or symlink
    ln -s /your/datasets/folder ./download/datasets
    ```
    Dataset YAML paths containing `${DATA_ROOT}` resolve from `DATA_ROOT`.
    Use `--data-root <path>` only when overriding `DATA_ROOT` for one command.

## Dataset Summary

| Dataset | Task | Eval Path |
|---------|------|-----------|
| [ImageNet ILSVRC2012](#imagenet-ilsvrc2012) | Image Classification | `${DATA_ROOT}/ILSVRC2012/val` |
| [DOTA v1](#dota-v1) | Oriented Object Detection | `${DATA_ROOT}/DOTAv1` |
| [Pascal VOC 2007/2012](#pascal-voc-20072012) | Object Detection / Semantic Segmentation | `${DATA_ROOT}/PascalVOC/VOCdevkit/VOC2012` |
| [COCO 2017](#coco-2017) | Instance Segmentation / Pose Estimation | `${DATA_ROOT}/COCO` |
| [Oxford-IIIT Pet](#oxford-iiit-pet) | Semantic Segmentation | `${DATA_ROOT}/Oxford-IIIT_Pet` |
| [Cityscapes](#cityscapes) | Semantic Segmentation | `${DATA_ROOT}/cityscapes` |
| [AFLW2000-3D](#aflw2000-3d) | 3D Face Landmark | `${DATA_ROOT}/AFLW20003D` |
| [Hand Keypoints](#hand-keypoints) | Hand Landmark | `${DATA_ROOT}/hand-keypoints` |
| [CelebA](#celeba) | Face Attribute | `${DATA_ROOT}/CelebA` |
| [LFW](#lfw-labeled-faces-in-the-wild) | Face Recognition | `${DATA_ROOT}/LFW` |
| [WiderFace](#widerface) | Face Detection | `${DATA_ROOT}/widerface` |
| [PETA](#peta-pedestrian-attribute) | Pedestrian Attribute | `${DATA_ROOT}/PETA` |
| [BSD100](#bsd100) | Super Resolution | `${DATA_ROOT}/BSD100/bicubic_4x` |
| [BSD68](#bsd68) | Image Denoising | `${DATA_ROOT}/BSD68` |
| [CBSD68](#cbsd68) | Image Denoising | `${DATA_ROOT}/CBSD68` |
| [LOL](#lol-low-light) | Low-Light Enhancement | `${DATA_ROOT}/LOL` |
| [NYU Depth v2](#nyu-depth-v2) | Depth Estimation | `${DATA_ROOT}/nyudepthv2` |
| [Market-1501](#market-1501) | Person Re-ID | `${DATA_ROOT}/Market1501` |

## Classification

### ImageNet ILSVRC2012

- **Download**: [ImageNet ILSVRC2012](https://image-net.org/download-images.php)
- **License**: Academic/Research use only
- **Eval path**: `${DATA_ROOT}/ILSVRC2012/val`

```text
ILSVRC2012/
├── val/
│   ├── n01440764/
│   │   └── .jpeg ×50
│   ├── n01443537/
│   │   └── .jpeg ×50
│   ├── n01484850/
│   │   └── .jpeg ×50
│   ├── n01491361/
│   │   └── .jpeg ×50
│   ├── n01494475/
│   │   └── .jpeg ×50
│   ├── n01496331/
│   │   └── .jpeg ×50
│   └── ... (994 more directories)
└── .txt ×3
```

## Object Detection

### DOTA v1

- **Download**: [DOTA v1](https://captain-whu.github.io/DOTA/dataset.html)
- **License**: Research use only
- **Eval path**: `${DATA_ROOT}/DOTAv1`

```text
DOTAv1/
├── images/
│   ├── test/
│   │   └── .jpg ×937
│   ├── train/
│   │   └── .jpg ×1411
│   └── val/
│       └── .jpg ×458
└── labels/
    ├── train/
    │   └── .txt ×1411
    ├── train_original/
    │   └── .txt ×1411
    ├── val/
    │   └── .txt ×458
    └── val_original/
        └── .txt ×458
```

### Pascal VOC 2007/2012

- **Download**: [Pascal VOC 2007/2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/)
- **License**: "VOC2007" challenge data

**Dataset classes:**

- **PascalVOC2007**: Object Detection
- **PascalVOC2012**: Semantic Segmentation

- **Eval path (VOC2012)**: `${DATA_ROOT}/PascalVOC/VOCdevkit/VOC2012`
- **Eval path (VOC2007)**: `${DATA_ROOT}/PascalVOC/VOCdevkit/VOC2007`

```text
PascalVOC/
└── VOCdevkit/
    ├── VOC2007/
    │   ├── Annotations/
    │   ├── ImageSets/
    │   ├── JPEGImages/
    │   ├── SegmentationClass/
    │   └── SegmentationObject/
    └── VOC2012/
        ├── Annotations/
        ├── ImageSets/
        ├── JPEGImages/
        ├── SegmentationClass/
        └── SegmentationObject/
```

## Segmentation

### COCO 2017

- **Download**: [COCO 2017](https://cocodataset.org/#download)
- **License**: Annotations under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); images remain under their original owners' rights and Flickr Terms of Use. See [COCO Terms of Use](https://cocodataset.org/#termsofuse).

**Dataset classes:**

- **COCO**: Instance Segmentation, Object Detection, Semantic Segmentation
- **COCOPose**: Pose Estimation

- **Eval path**: `${DATA_ROOT}/COCO`

```text
COCO/
├── annotations/
│   ├── instances_val2017.json
│   └── person_keypoints_val2017.json
└── val2017/
    └── .jpg ×5000
```

### Oxford-IIIT Pet

- **Download**: [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- **License**: CC BY-SA 4.0 (free for research and commercial use)
- **Eval path**: `${DATA_ROOT}/Oxford-IIIT_Pet`

```text
Oxford-IIIT_Pet/
├── annotations/
│   ├── trimaps/
│   │   └── .png ×14780
│   ├── xmls/
│   │   └── .xml ×3686
│   └── .txt ×3,  ×2
└── images/
    └── .jpg ×7390, .mat ×3
```

### Cityscapes

- **Download**: [Cityscapes](https://www.cityscapes-dataset.com/register/)
- **License**: Academic/Research use only
- **Eval path**: `${DATA_ROOT}/cityscapes`

```text
cityscapes/
├── gtFine/
│   └── val/
│       ├── frankfurt/
│       ├── lindau/
│       └── munster/
├── leftImg8bit/
│   └── val/
│       ├── frankfurt/
│       ├── lindau/
│       └── munster/
└── .txt ×2
```

## Pose / Landmark

### AFLW2000-3D

- **Download**: [AFLW2000-3D](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm)
- **License**: Research use only
- **Eval path**: `${DATA_ROOT}/AFLW20003D`

```text
AFLW20003D/
├── Code/
│   └── .pkl ×3, .npy ×2, .txt ×1
└── .jpg ×2000, .mat ×2000
```

### Hand Keypoints

- **License**: Custom dataset
- **Eval path**: `${DATA_ROOT}/hand-keypoints`

```text
hand-keypoints/
├── images/
│   ├── train/
│   │   └── .jpg ×18776
│   └── val/
│       └── .jpg ×7992
├── labels/
│   ├── train/
│   │   └── .txt ×18776
│   └── val/
│       └── .txt ×7992
└── .yaml ×1
```

## Face

### CelebA

- **Download**: [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- **License**: Non-commercial research only
- **Eval path**: `${DATA_ROOT}/CelebA`

```text
CelebA/
├── img_align_celeba/
│   └── .jpg ×19962
└── .csv ×4
```

### LFW (Labeled Faces in the Wild)

- **Download**: [LFW (Labeled Faces in the Wild)](http://vis-www.cs.umass.edu/lfw/)
- **License**: Research use only
- **Eval path**: `${DATA_ROOT}/LFW`

```text
LFW/
├── lfw-deepfunneled/
│   ├── Aaron_Eckhart/
│   │   └── .jpg ×1
│   ├── Aaron_Guiel/
│   │   └── .jpg ×1
│   ├── Aaron_Patterson/
│   │   └── .jpg ×1
│   ├── Aaron_Peirsol/
│   │   └── .jpg ×4
│   ├── Aaron_Pena/
│   │   └── .jpg ×1
│   ├── Aaron_Sorkin/
│   │   └── .jpg ×2
│   └── ... (5743 more directories)
└── .csv ×10
```

### WiderFace

- **Download**: [WiderFace](http://shuoyang1213.me/WIDERFACE/)
- **License**: Non-commercial research only
- **Eval path**: `${DATA_ROOT}/widerface`

```text
widerface/
├── eval_tools/
│   ├── ground_truth/
│   │   └── .mat ×4
│   ├── plot/
│   │   ├── baselines/
│   │   ├── figure/
│   │   └── .m ×4,  ×1
│   └── .m ×6,  ×1
├── wider_face_split/
│   └── .txt ×4, .mat ×3
├── WIDER_train/
├── WIDER_val/
│   └── images/
│       ├── 0--Parade/
│       ├── 1--Handshaking/
│       ├── 10--People_Marching/
│       ├── 11--Meeting/
│       ├── 12--Group/
│       ├── 13--Interview/
│       └── ... (55 more directories)
└── .zip ×1
```

## Person

### PETA (PEdesTrian Attribute)

- **Download**: [PETA (PEdesTrian Attribute)](https://mmlab.ie.cuhk.edu.hk/projects/PETA.html)
- **License**: Research use only
- **Eval path**: `${DATA_ROOT}/PETA`

```text
PETA/
├── images/
│   └── .png ×19000
└── .mat ×1,  ×1
```

## Image Restoration

### BSD100

- **Download**: [BSD100](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)
- **License**: BSD license (free for research and commercial use)
- **Eval path**: `${DATA_ROOT}/BSD100/bicubic_4x`

```text
BSD100/
├── bicubic_2x/
│   ├── HR/
│   │   └── .png ×100
│   └── LR/
│       └── .png ×100
├── bicubic_3x/
│   ├── HR/
│   │   └── .png ×100
│   └── LR/
│       └── .png ×100
└── bicubic_4x/
    ├── HR/
    │   └── .png ×100
    └── LR/
        └── .png ×100
```

### BSD68

- **Download**: [BSD68](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)
- **License**: BSD license (free for research and commercial use)
- **Eval path**: `${DATA_ROOT}/BSD68`

```text
BSD68/
└── .png ×68
```

### CBSD68

- **Download**: [CBSD68](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)
- **License**: BSD license (free for research and commercial use)
- **Eval path**: `${DATA_ROOT}/CBSD68`

```text
CBSD68/
└── .png ×68
```

### LOL (Low-Light)

- **Download**: [LOL (Low-Light)](https://daooshee.github.io/BMVC2018website/)
- **License**: Research use only
- **Eval path**: `${DATA_ROOT}/LOL`

```text
LOL/
├── eval15/
│   ├── high/
│   │   └── .png ×15
│   └── low/
│       └── .png ×15
└── our485/
    ├── high/
    │   └── .png ×485
    └── low/
        └── .png ×485
```

## Depth

### NYU Depth v2

- **Download**: [NYU Depth v2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
- **License**: Research use only
- **Eval path**: `${DATA_ROOT}/nyudepthv2`

```text
nyudepthv2/
└── val/
    └── official/
        └── .h5 ×654
```

## Other

### Market-1501

- **Download**: [Market-1501](https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html)
- **License**: Research use only
- **Eval path**: `${DATA_ROOT}/Market1501`

```text
Market1501/
├── bounding_box_test/
│   └── .jpg ×19732
└── query/
    └── .jpg ×3368
```
