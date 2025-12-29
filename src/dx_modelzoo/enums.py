from enum import StrEnum


class SessionType(StrEnum):
    onnxruntime = "OnnxRuntime"
    simulator = "Simulator"
    dxruntime = "DxRuntime"


class EvaluationType(StrEnum):
    image_classification = "ImageClassification"
    coco = "ObjectDection_COCO"
    segmentation = "ImageSegmentation"
    voc = "ObjectDetection_VOC2007"
    bsd68 = "ImageDenosing_BSD68"
    bsd100 = "ImageDenosing_BSD100"
    widerface = "FaceDetection"
    depth_estimation = "DepthEstimation"
    instance_segmentation = "InstanceSegmentation"
    zeroshot_classification = "ZeroShotClassification"
    coco_pose = "PoseEstimation"

    def metric(self) -> str:
        if self.value in {EvaluationType.image_classification, EvaluationType.zeroshot_classification}:
            return "TopK1, TopK5"
        elif self.value == EvaluationType.coco:
            return "mAP, mAP50"
        elif self.value == EvaluationType.voc:
            return "mAP50"
        elif self.value == EvaluationType.segmentation:
            return "mIoU"
        elif self.value == EvaluationType.widerface:
            return "AP"
        elif self.value in {EvaluationType.bsd68, EvaluationType.bsd100}:
            return "PSNR, SSIM"
        elif self.value == EvaluationType.depth_estimation:
            return "RMSE"
        elif self.value == EvaluationType.instance_segmentation:
            return "mAP"
        elif self.value == EvaluationType.coco_pose:
            return "mAP, mAP50"
        else:
            raise ValueError(f"Invalid Evaluation Type value. {self.value}")


class DatasetType(StrEnum):
    imagenet = "ImageNet"
    coco = "COCO"
    voc_seg = "VOCSegmentation"
    voc_od = "VOC2007Detection"
    bsd68 = "BSD68"
    bsd100 = "BSD100"
    city = "CitySpace"
    widerface = "WiderFace"
    nyu = "NYU"
    coco_pose = "COCOPose"
