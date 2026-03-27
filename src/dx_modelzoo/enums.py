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
    cbsd68 = "ImageDenosing_CBSD68"
    bsd100 = "ImageDenosing_BSD100"
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
        elif self.value in {EvaluationType.bsd68, EvaluationType.cbsd68, EvaluationType.bsd100}:
            return "PSNR, SSIM"
        elif self.value == EvaluationType.depth_estimation:
            return "RMSE"
        elif self.value == EvaluationType.instance_segmentation:
            return "mAP"
        elif self.value == EvaluationType.coco_pose:
            return "mAP, mAP50"
        elif self.value == EvaluationType.obb:
            return "mAP, mAP50"
        elif self.value == EvaluationType.lfw:
            return "Accuracy"
        elif self.value == EvaluationType.zeroshot_instance_segmentation:
            return "AR@10, AR@100, AR@1000"
        elif self.value == EvaluationType.hand_landmark:
            return "MNAE"
        elif self.value == EvaluationType.face_attribute:
            return "AverageAccuracy"
        elif self.value == EvaluationType.person_attribute:
            return "mA"
        elif self.value == EvaluationType.person_reid:
            return "Rank1, mAP"
        elif self.value == EvaluationType.oxford_pet:
            return "mIoU"
        elif self.value == EvaluationType.face_landmark:
            return "NME"
        elif self.value == EvaluationType.lol:
            return "PSNR, SSIM"
        else:
            raise ValueError(f"Invalid Evaluation Type value. {self.value}")


class DatasetType(StrEnum):
    imagenet = "ImageNet"
    coco = "COCO"
    voc_seg = "VOCSegmentation"
    voc_od = "VOC2007Detection"
    bsd68 = "BSD68"
    cbsd68 = "CBSD68"
    bsd100 = "BSD100"
    city = "CityScapes"
    widerface = "WiderFace"
    nyu = "NYU"
    coco_pose = "COCOPose"
    dotav1 = "DOTAv1"
    lfw = "LFW"
    hand_landmark = "HandLandmark"
    celeba = "CelebA"
    peta = "PETA"
    market1501 = "Market1501"
    oxford_pet = "OxfordPet"
    aflw2000_3d = "AFLW20003D"
    lol = "LOL"
