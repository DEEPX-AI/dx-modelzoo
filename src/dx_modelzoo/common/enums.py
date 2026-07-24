from __future__ import annotations

from enum import Enum


class DeviceType(str, Enum):
    GPU = "gpu"
    CPU = "cpu"
    NPU = "npu"

    def is_npu(self) -> bool:
        return self == DeviceType.NPU


class SessionType(str, Enum):
    onnxruntime = "OnnxRuntime"
    simulator = "Simulator"
    dxruntime = "DxRuntime"


class EvaluationType(str, Enum):
    image_classification = "ImageClassification"
    coco = "ObjectDetection_COCO"
    segmentation = "ImageSegmentation"
    voc = "ObjectDetection_VOC2007"
    bsd68 = "ImageDenoising_BSD68"
    cbsd68 = "ImageDenoising_CBSD68"
    bsd100 = "ImageDenoising_BSD100"
    widerface = "FaceDetection"
    depth_estimation = "DepthEstimation"
    instance_segmentation = "InstanceSegmentation"
    zeroshot_classification = "ZeroShotClassification"
    coco_pose = "PoseEstimation"
    obb = "OrientedObjectDetection"
    lfw = "FaceVerification"
    zeroshot_instance_segmentation = "ZeroShotInstanceSegmentation"
    hand_landmark = "HandLandmark"
    face_attribute = "FaceAttribute"
    person_attribute = "PersonAttributeRecognition"
    person_reid = "PersonReID"
    oxford_pet = "OxfordPetSegmentation"
    face_landmark = "FaceLandmark"
    lol = "LowLightEnhancement"

    def metric(self) -> str:
        METRIC_MAP = {
            "ImageClassification": "TopK1, TopK5",
            "ZeroShotClassification": "TopK1, TopK5",
            "ObjectDetection_COCO": "mAP, mAP50",
            "ObjectDetection_VOC2007": "mAP50",
            "ImageSegmentation": "mIoU",
            "OxfordPetSegmentation": "mIoU",
            "FaceDetection": "AP",
            "ImageDenoising_BSD68": "PSNR, SSIM",
            "ImageDenoising_CBSD68": "PSNR, SSIM",
            "ImageDenoising_BSD100": "PSNR, SSIM",
            "DepthEstimation": "RMSE",
            "InstanceSegmentation": "mAP",
            "PoseEstimation": "mAP, mAP50",
            "OrientedObjectDetection": "mAP, mAP50",
            "FaceVerification": "Accuracy",
            "ZeroShotInstanceSegmentation": "AR@10, AR@100, AR@1000",
            "HandLandmark": "MNAE",
            "FaceAttribute": "AverageAccuracy",
            "PersonAttributeRecognition": "mA",
            "PersonReID": "Rank1, mAP",
            "FaceLandmark": "NME",
            "LowLightEnhancement": "PSNR, SSIM",
        }
        return METRIC_MAP.get(self.value, "Unknown")


class DatasetType(str, Enum):
    BSD68 = "BSD68"
    BSD100 = "BSD100"
    CBSD68 = "CBSD68"
    CelebA = "CelebA"
    Cityscapes = "Cityscapes"
    COCO = "COCO"
    COCOPose = "COCOPose"
    DOTAv1 = "DOTAv1"
    AFLW20003D = "AFLW20003D"
    HandKeypoints = "HandKeypoints"
    LFW = "LFW"
    LOL = "LOL"
    Market1501 = "Market1501"
    NYUDepthv2 = "NYUDepthv2"
    OxfordIIITPet = "OxfordIIITPet"
    PETA = "PETA"
    PascalVOC2012 = "PascalVOC2012"
    PascalVOC2007 = "PascalVOC2007"
    WiderFace = "WiderFace"
    ILSVRC2012 = "ILSVRC2012"
