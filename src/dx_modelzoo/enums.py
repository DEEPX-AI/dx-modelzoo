from enum import StrEnum


class SessionType(StrEnum):
    onnxruntime = "OnnxRuntime"
    simulator = "Simulator"
    dxruntime = "DxRuntime"


class EvaluationType(StrEnum):
    image_classification = "ImageClassification"
    coco = "ObjectDection"
    segmentation = "ImageSegmentation"
    voc = "ObjectDetection"
    bsd68 = "ImageDenosing"
    widerface = "FaceDetection"

    def metric(self) -> str:
        if self.value == EvaluationType.image_classification:
            return "TopK1, TopK5"
        elif self.value == EvaluationType.coco:
            return "mAP, mAP50"
        elif self.value == EvaluationType.voc:
            return "mAP50"
        elif self.value == EvaluationType.segmentation:
            return "mIoU"
        elif self.value == EvaluationType.widerface:
            return "AP"
        else:
            raise ValueError(f"Invalid Evaluation Type value. {self.value}")


class DatasetType(StrEnum):
    imagenet = "ImageNet"
    coco = "COCO"
    voc_seg = "VOCSegmentation"
    voc_od = "VOC2007Detection"
    bsd68 = "BSD68"
    city = "CitySpace"
    widerface = "WiderFace"
