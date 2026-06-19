# Datasets

Configuration reference for all evaluation datasets supported by DX-ModelZoo.
Default dataset root is `$DATA_ROOT` (set via environment variable or YAML `dataset.eval_path`).

!!! tip "Quick Setup"
    Create symlinks to your actual dataset locations:
    ```bash
    export DATA_ROOT=/your/datasets/folder
    # or symlink
    ln -s /your/datasets/folder ./download/datasets
    ```

---

## AFLW2000-3D

- **Download**: [AFLW2000-3D](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm)
- **License**: Research use only
- **Eval path**: `${DATA_ROOT}/AFLW20003D`

```text
AFLW20003D/
├── Code/
│   └── .pkl ×3, .npy ×2, .txt ×1
└── .jpg ×2000, .mat ×2000
```

```bash
dxmz eval <ModelName> --profile onnx --data-dir $DATA_ROOT/AFLW20003D
```

---

## BSD100

- **Download**: [BSD100](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)
- **License**: BSD license (free for research and commercial use)
- **Eval path**: `${DATA_ROOT}/BSD100`

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

```bash
dxmz eval <ModelName> --profile onnx --data-dir $DATA_ROOT/BSD100
```

---

## BSD68

- **Download**: [BSD68](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)
- **License**: BSD license (free for research and commercial use)
- **Eval path**: `${DATA_ROOT}/BSD68`

```text
BSD68/
└── .png ×68
```

```bash
dxmz eval <ModelName> --profile onnx --data-dir $DATA_ROOT/BSD68
```

---

## CBSD68

- **Download**: [CBSD68](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)
- **License**: BSD license (free for research and commercial use)
- **Eval path**: `${DATA_ROOT}/CBSD68`

```text
CBSD68/
└── .png ×68
```

```bash
dxmz eval <ModelName> --profile onnx --data-dir $DATA_ROOT/CBSD68
```

---

## COCO 2017

- **Download**: [COCO 2017](https://cocodataset.org/#download)
- **License**: CC BY 4.0 (free for research and commercial use)

**Dataset classes:**

- **COCO**: Instance Segmentation, Object Detection, Semantic Segmentation
- **COCOPose**: Pose Estimation

- **Eval path**: `${DATA_ROOT}/COCO/official`

```text
COCO/
└── official/
    ├── annotations/
    │   └── .json ×6
    ├── images/
    │   └── val2017/
    ├── labels/
    │   ├── train2017/
    │   ├── val2017/
    │   └── .cache ×1
    ├── val2017/
    │   └── .jpg ×5000
    └── .txt ×5,  ×1
```

```bash
dxmz eval <ModelName> --profile onnx --data-dir $DATA_ROOT/COCO/official
```

---

## CelebA

- **Download**: [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- **License**: Non-commercial research only
- **Eval path**: `${DATA_ROOT}/CelebA`

```text
CelebA/
├── img_align_celeba/
│   └── .jpg ×19962
└── .csv ×4
```

```bash
dxmz eval <ModelName> --profile onnx --data-dir $DATA_ROOT/CelebA
```

---

## DOTA v1

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

```bash
dxmz eval <ModelName> --profile onnx --data-dir $DATA_ROOT/DOTAv1
```

---

## ImageNet ILSVRC2012

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

```bash
dxmz eval <ModelName> --profile onnx --data-dir $DATA_ROOT/ILSVRC2012/val
```

---

## LFW (Labeled Faces in the Wild)

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

```bash
dxmz eval <ModelName> --profile onnx --data-dir $DATA_ROOT/LFW
```

---

## LOL (Low-Light)

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

```bash
dxmz eval <ModelName> --profile onnx --data-dir $DATA_ROOT/LOL
```

---

## Oxford-IIIT Pet

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

```bash
dxmz eval <ModelName> --profile onnx --data-dir $DATA_ROOT/Oxford-IIIT_Pet
```

---

## PETA (PEdesTrian Attribute)

- **Download**: [PETA (PEdesTrian Attribute)](https://mmlab.ie.cuhk.edu.hk/projects/PETA.html)
- **License**: Research use only
- **Eval path**: `${DATA_ROOT}/PETA`

```text
PETA/
├── images/
│   └── .png ×19000
└── .mat ×1,  ×1
```

```bash
dxmz eval <ModelName> --profile onnx --data-dir $DATA_ROOT/PETA
```

---

## Pascal VOC 2007/2012

- **Download**: [Pascal VOC 2007/2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/)
- **License**: "VOC2007" challenge data

**Dataset classes:**

- **PascalVOC2007**: Object Detection
- **PascalVOC2012**: Semantic Segmentation

- **Eval path**: `${DATA_ROOT}/PascalVOC/VOCdevkit/VOC2012`
- **Eval path**: `${DATA_ROOT}/PascalVOC/VOCdevkit/VOC2007`

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

```bash
dxmz eval <ModelName> --profile onnx --data-dir $DATA_ROOT/PascalVOC/VOCdevkit/VOC2012
```

---

## Cityscapes

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

```bash
dxmz eval <ModelName> --profile onnx --data-dir $DATA_ROOT/cityscapes
```

---

## Hand Keypoints

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

```bash
dxmz eval <ModelName> --profile onnx --data-dir $DATA_ROOT/hand-keypoints
```

---

## NYU Depth v2

- **Download**: [NYU Depth v2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
- **License**: Research use only
- **Eval path**: `${DATA_ROOT}/nyudepthv2`

```text
nyudepthv2/
└── val/
    └── official/
        └── .h5 ×654
```

```bash
dxmz eval <ModelName> --profile onnx --data-dir $DATA_ROOT/nyudepthv2
```

---

## WiderFace

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

```bash
dxmz eval <ModelName> --profile onnx --data-dir $DATA_ROOT/widerface
```

---

## Market-1501

- **Download**: [Market-1501](https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html)
- **License**: Research use only
```bash
dxmz eval <ModelName> --profile onnx
```

---
