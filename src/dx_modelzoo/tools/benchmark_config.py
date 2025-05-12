import os
from dx_modelzoo.enums import DatasetType

class Modelzoo_config:
    def __init__(self, args=None): 
        
        # Path setting 
        self.MODEL_LOCAL_PATH = "./open_models"
        self.COMPILER_VER = '1_32_4'  # AWS folder

        datasets_path = args.data_dir if hasattr(args, "data_dir") else "./datasets"

        # Evaluation config
        self.data_dir_dict = {
            DatasetType.imagenet: os.path.join(datasets_path, "ILSVRC2012", "val"),
            DatasetType.coco: os.path.join(datasets_path, "COCO", "official"),
            DatasetType.voc_od: os.path.join(datasets_path, "PascalVOC", "VOCdevkit", "VOC2007"),
            DatasetType.bsd68: os.path.join(datasets_path, "BSD68"),
            DatasetType.city: os.path.join(datasets_path, "cityscapes"),
            DatasetType.voc_seg: os.path.join(datasets_path, "PascalVOC", "VOCdevkit", "VOC2012"),
            DatasetType.widerface: os.path.join(datasets_path, "widerface"),
        }
        
        if args is not None:
            self.validate_data_dirs()
        
        # Registered Model list config
        self.model_list_by_task = {
            'image_classfication': [
                'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152',
                'AlexNet', 'VGG11', 'VGG13', 'VGG13BN', 'VGG11BN', 'VGG19BN',
                'MobileNetV1', 'MobileNetV2', 'MobileNetV3Small', 'MobileNetV3Large',
                'SqueezeNet1_0', 'SqueezeNet1_1', 'EfficientNetB2', 'EfficientNetV2S',
                'RegNetX400MF', 'RegNetX800MF', 'RegNetY200MF', 'RegNetY400MF', 'RegNetY800MF',
                'ResNeXt50_32x4d', 'ResNeXt26_32x4d',
                'DenseNet121', 'DenseNet161', 'HarDNet39DS', 'WideResNet50_2', 'WideResNet101_2',
                'OSNet0_25', 'OSNet0_5', 'RepVGGA1'
            ],
            'object_detection': [
                'YoloV3', 'YoloV5N', 'YoloV5S', 'YoloV5M', 'YoloV5L', 'YoloV7', 'YoloV7E6',
                'YoloV7Tiny', 'YoloV8L', 'YoloV9C', 'YoloV9S', 'YoloV9T', 'YoloXS',
                'SSDMV1', 'SSDMV2Lite'
            ],
            'face_id': [
                'YOLOv5s_Face', 'YOLOv5m_Face', 'YOLOv7s_Face', 'YOLOv7_Face',
                'YOLOv7_TTA_Face', 'YOLOv7_W6_Face', 'YOLOv7_W6_TTA_Face'
            ],
            'semantic_segmentation': [
                'DeepLabV3PlusMobilenet', 'DeepLabV3PlusDRN', 'DeepLabV3PlusMobileNetV2',
                'DeepLabV3PlusResNet101', 'DeepLabV3PlusResNet50', 'DeepLabV3PlusResnet',
                'BiSeNetV1', 'BiSeNetV2'
            ],
            'image_denoising': ['DnCNN_15', 'DnCNN_25', 'DnCNN_50']
        }
    
    def validate_data_dirs(self):
        for dataset, path in self.data_dir_dict.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"[ERROR] Dataset path not found for {dataset.value}: {path}")
