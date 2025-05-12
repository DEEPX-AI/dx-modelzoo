from dx_modelzoo.models.face_detection.yolo import (
    YOLOv5m_Face,
    YOLOv5s_Face,
    YOLOv7_Face,
    YOLOv7_TTA_Face,
    YOLOv7_W6_Face,
    YOLOv7_W6_TTA_Face,
    YOLOv7s_Face,
)
from dx_modelzoo.models.image_classification.alexnet import AlexNet
from dx_modelzoo.models.image_classification.densenet import DenseNet121, DenseNet161
from dx_modelzoo.models.image_classification.efficientnet import EfficientNetB2, EfficientNetV2S
from dx_modelzoo.models.image_classification.hardnet import HarDNet39DS
from dx_modelzoo.models.image_classification.mobilenet import (
    MobileNetV1,
    MobileNetV2,
    MobileNetV3Large,
    MobileNetV3Small,
)
from dx_modelzoo.models.image_classification.osnet import OSNet0_5, OSNet0_25
from dx_modelzoo.models.image_classification.regnet import (
    RegNetX400MF,
    RegNetX800MF,
    RegNetY200MF,
    RegNetY400MF,
    RegNetY800MF,
)
from dx_modelzoo.models.image_classification.repvgg import RepVGGA1
from dx_modelzoo.models.image_classification.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from dx_modelzoo.models.image_classification.resnext import ResNeXt26_32x4d, ResNeXt50_32x4d
from dx_modelzoo.models.image_classification.squeezenet import SqueezeNet1_0, SqueezeNet1_1
from dx_modelzoo.models.image_classification.vgg import VGG11, VGG11BN, VGG13, VGG13BN, VGG19BN
from dx_modelzoo.models.image_classification.wideresnet import WideResNet50_2, WideResNet101_2
from dx_modelzoo.models.image_denoising.dncnn import DnCNN_15, DnCNN_25, DnCNN_50
from dx_modelzoo.models.image_segmentation.bisenet import BiSeNetV1, BiSeNetV2
from dx_modelzoo.models.image_segmentation.deeplab import (
    DeepLabV3PlusDRN,
    DeepLabV3PlusMobilenet,
    DeepLabV3PlusMobileNetV2,
    DeepLabV3PlusResnet,
    DeepLabV3PlusResNet50,
    DeepLabV3PlusResNet101,
)
from dx_modelzoo.models.object_dection.ssd import SSDMV1, SSDMV2Lite
from dx_modelzoo.models.object_dection.yolo import (
    YoloV3,
    YoloV5L,
    YoloV5M,
    YoloV5N,
    YoloV5S,
    YoloV7,
    YoloV7E6,
    YoloV7Tiny,
    YoloV8L,
    YoloV8M,
    YoloV8N,
    YoloV8S,
    YoloV8X,
    YoloV9C,
    YoloV9S,
    YoloV9T,
    YoloXS,
)

MODEL_DICT = {
    ResNet18.__name__: ResNet18,
    YoloV5N.__name__: YoloV5N,
    DeepLabV3PlusMobilenet.__name__: DeepLabV3PlusMobilenet,
    ResNet34.__name__: ResNet34,
    ResNet50.__name__: ResNet50,
    ResNet101.__name__: ResNet101,
    ResNet152.__name__: ResNet152,
    YoloV5S.__name__: YoloV5S,
    YoloV5M.__name__: YoloV5M,
    YoloV5L.__name__: YoloV5L,
    DeepLabV3PlusDRN.__name__: DeepLabV3PlusDRN,
    DeepLabV3PlusMobileNetV2.__name__: DeepLabV3PlusMobileNetV2,
    DeepLabV3PlusResNet101.__name__: DeepLabV3PlusResNet101,
    DeepLabV3PlusResNet50.__name__: DeepLabV3PlusResNet50,
    DeepLabV3PlusResnet.__name__: DeepLabV3PlusResnet,
    ResNeXt50_32x4d.__name__: ResNeXt50_32x4d,
    ResNeXt26_32x4d.__name__: ResNeXt26_32x4d,
    RegNetX400MF.__name__: RegNetX400MF,
    RegNetX800MF.__name__: RegNetX800MF,
    RegNetY200MF.__name__: RegNetY200MF,
    RegNetY400MF.__name__: RegNetY400MF,
    RegNetY800MF.__name__: RegNetY800MF,
    DenseNet121.__name__: DenseNet121,
    DenseNet161.__name__: DenseNet161,
    EfficientNetB2.__name__: EfficientNetB2,
    EfficientNetV2S.__name__: EfficientNetV2S,
    HarDNet39DS.__name__: HarDNet39DS,
    MobileNetV1.__name__: MobileNetV1,
    MobileNetV2.__name__: MobileNetV2,
    SqueezeNet1_0.__name__: SqueezeNet1_0,
    SqueezeNet1_1.__name__: SqueezeNet1_1,
    VGG11BN.__name__: VGG11BN,
    VGG19BN.__name__: VGG19BN,
    WideResNet101_2.__name__: WideResNet101_2,
    WideResNet50_2.__name__: WideResNet50_2,
    AlexNet.__name__: AlexNet,
    VGG11.__name__: VGG11,
    VGG13.__name__: VGG13,
    VGG13BN.__name__: VGG13BN,
    MobileNetV3Large.__name__: MobileNetV3Large,
    MobileNetV3Small.__name__: MobileNetV3Small,
    OSNet0_25.__name__: OSNet0_25,
    OSNet0_5.__name__: OSNet0_5,
    RepVGGA1.__name__: RepVGGA1,
    YoloXS.__name__: YoloXS,
    YoloV3.__name__: YoloV3,
    YoloV7.__name__: YoloV7,
    YoloV7E6.__name__: YoloV7E6,
    YoloV7Tiny.__name__: YoloV7Tiny,
    YoloV8L.__name__: YoloV8L,
    YoloV9C.__name__: YoloV9C,
    YoloV9S.__name__: YoloV9S,
    YoloV9T.__name__: YoloV9T,
    SSDMV1.__name__: SSDMV1,
    SSDMV2Lite.__name__: SSDMV2Lite,
    DnCNN_15.__name__: DnCNN_15,
    DnCNN_25.__name__: DnCNN_25,
    DnCNN_50.__name__: DnCNN_50,
    BiSeNetV1.__name__: BiSeNetV1,
    BiSeNetV2.__name__: BiSeNetV2,
    YOLOv5s_Face.__name__: YOLOv5s_Face,
    YOLOv5m_Face.__name__: YOLOv5m_Face,
    YOLOv7s_Face.__name__: YOLOv7s_Face,
    YOLOv7_Face.__name__: YOLOv7_Face,
    YOLOv7_TTA_Face.__name__: YOLOv7_TTA_Face,
    YOLOv7_W6_Face.__name__: YOLOv7_W6_Face,
    YOLOv7_W6_TTA_Face.__name__: YOLOv7_W6_TTA_Face,
    YoloV8X.__name__: YoloV8X,
    YoloV8M.__name__: YoloV8M,
    YoloV8N.__name__: YoloV8N,
    YoloV8S.__name__: YoloV8S,
}
