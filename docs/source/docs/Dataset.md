# Dataset

In DeepX Open ModelZoo, models are benchmarked on public datasets for different tasks. Default dataset folder points to `./datasets`. We recommend creating soft links to your actual dataset locations:

```
ls -n /your/datasets/folder ./datasets
```

## ImageNet (ILSVRC2012)

- Purpose: Image classification evaluation.
- Available Models:
  - 'DenseNet121', 'DenseNet161', 'DnCNN_15', 'DnCNN_25', 'DnCNN_50', 'EfficientNetB2', 'EfficientNetV2S', 'HarDNet39DS', 'MobileNetV1', 'MobileNetV2', 'MobileNetV3Large', 'MobileNetV3Small', 'AlexNet'
- Download : [ImageNet](http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar).

* Dataset structure:
  ```
  ILSVRC2012/
  └── val/
      ├── n01440764/
      |     ├── ILSVRC2012_val_*.JPEG
      |     ├── ...
      ...
  ```

- Usage example:
  ```
  dxmz eval <Classification Model Name> --dxrt <dxnn file path> --data_dir ./datasets/ILSVRC2012/val
  ```

## COCO2017

- Purpose: Object detection evaluation.
- Available Models:

  - 'YoloV3', 'YoloV5L', 'YoloV5M', 'YoloV5N', 'YoloV5S', 'YoloV7', 'YoloV7E6', 'YoloV7Tiny', 'YoloV8L', 'YoloV9C', 'YoloV9S', 'YoloV9T', 'YoloXS'

- Download: [COCO2017](http://images.cocodataset.org/zips/val2017.zip).
- Dataset structure:
  ```
  COCO/official
          ├── images/
          |   ├── val2017/
          │       ├── *.jpg
          │       ├── ...
          ├── annotations/
          │    └── instances_val2017.json
          ...
  ```
- Usage Example:
  ```
  dxmz eval <Object Detection Model Name> --dxrt <dxnn file path> --data_dir ./datasets/COCO/official
  ```

## PascalVOC - VOC2007

- Purpose: Object detection evaluation.
- Available Models:
  - 'SSDMV1', 'SSDMV2Lite'
- Download: ([VOC2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html)).
- Dataset Structure:
  ```
  PascalVOC/VOCdevkit/VOC2007/
                        ├── ImageSets/Main/test.txt
                        ├── JPEGImages/
                        │   ├── *.jpg
                        |   ├── ...
                        ├── Annotations/
                        │   ├── *.xml
                        |   ├── ...
                        ...
  ```
- Usage Example:
  ```
  dxmz eval <Object Detection Model Name> --dxrt <dxnn file path> --data_dir ./datasets/PascalVOC/VOCdevkit/VOC2007
  ```

## PascalVOC - VOC2012

- Purpose: Semantic segmentation evaluation.
- Available Model:
  - 'DeepLabV3PlusMobilenet'
- Download: ([VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit)).
- Dataset Structure:
  ```
  PascalVOC/VOCdevkit/VOC2012/
                        ├── JPEGImages/
                        │   ├── *.jpg
                        │   ├── ...
                        ├── SegmentationClass/
                        │   ├── *.png
                        │   ├── ...
                        ├── ImageSets/
                        │   ├── Segmentation/
                        │   │   ├── val.txt
                        ...
  ```
- Usage Example:
  ```
  dxmz eval <Segmentation Model Name> --dxrt <dxnn file path> --data_dr ./datasets/PascalVOC/VOCdevkit/VOC2012
  ```

## Cityscapes

- Purpose: Semantic segmentation evaluation.
- Available Models:
  - 'BiSeNetV1', 'BiSeNetV2'
- Download: ([Cityscapes](https://www.cityscapes-dataset.com/)).
- Dataset structure:
  ```
  cityscapes/
      ├── leftImg8bit/val/[city]/*.png
      ├── gtFine/val/[city]/*.png
      └── val.txt
  ```
- Usage Example:

```
dxmz eval <Segmentation Model Name> --dxrt <dxnn file path> --data_dir ./datasets/cityscapes
```

## WiderFace

- Purpose: Face detection evaluation
- Available Models:
  - 'YOLOv5m_Face', 'YOLOv5s_Face', 'YOLOv7_Face', 'YOLOv7_TTA_Face', 'YOLOv7_W6_Face', 'YOLOv7_W6_TTA_Face', 'YOLOv7s_Face'
- Download: ([WiderFace](http://shuoyang1213.me/WIDERFACE/))
- Dataset Structure:
  ```
  widerface/
  ├── WIDER_val/images/[event]/*.jpg
  └── eval_tools/ground_truth/
        ├── wider_face_val.mat
        ├── wider_[difficulty]_val.mat
  ```
- Usage Example:
  ```
  dxmz eval <Face ID Model Name> --dxrt <dxnn file path> --data_dir ./datasets/widerface
  ```

## BSD68

- Purpose: Image denoising evaluation
- Available Models:
  - ...
- Download ([BSD68](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)).
- Dataset Structure:
  ```
  BSD68/
  ├── *.png
  └──
  ```
- Usage Example:
  ```
  dxmz eval <Image Denoising Model Name> --dxrt <dxnn file path> --data_dir ./datasets/BSD68
  ```
