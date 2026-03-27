from dx_modelzoo.enums import EvaluationType
from dx_modelzoo.evaluator.bsd68_evaluator import BSD68Evaluator
from dx_modelzoo.evaluator.bsd100_evaluator import BSD100Evaluator
from dx_modelzoo.evaluator.clip_evaluator import CLIPEvaluator
from dx_modelzoo.evaluator.coco_evaluator import COCOEvaluator
from dx_modelzoo.evaluator.coco_pose_evaluator import COCOPoseEvaluator
from dx_modelzoo.evaluator.depth_estimation_evaluator import DepthEstimationEvaluator
from dx_modelzoo.evaluator.face_attribute_evaluator import FaceAttributeEvaluator
from dx_modelzoo.evaluator.face_landmark_evaluator import FaceLandmarkEvaluator
from dx_modelzoo.evaluator.hand_landmark_evaluator import HandLandmarkEvaluator
from dx_modelzoo.evaluator.ic_evaluator import ICEvaluator
from dx_modelzoo.evaluator.instance_seg_evaluator import InstanceSegEvaluator
from dx_modelzoo.evaluator.lfw_evaluator import LFWEvaluator
from dx_modelzoo.evaluator.lol_evaluator import LOLEvaluator
from dx_modelzoo.evaluator.obb_evaluator import OBBEvaluator
from dx_modelzoo.evaluator.person_attribute_evaluator import PersonAttributeEvaluator
from dx_modelzoo.evaluator.person_reid_evaluator import PersonReIDEvaluator
from dx_modelzoo.evaluator.segmentation_evaluator import SegentationEvaluator
from dx_modelzoo.evaluator.voc_evaluator import VOC2007DetectionEvaluator
from dx_modelzoo.evaluator.widerface_evaluator import WiderFaceEvaluator
from dx_modelzoo.evaluator.zeroshot_instance_seg_evaluator import ZeroShotInstanceSegEvaluator

EVAL_DICT = {
    EvaluationType.image_classification: ICEvaluator,
    EvaluationType.coco: COCOEvaluator,
    EvaluationType.segmentation: SegentationEvaluator,
    EvaluationType.voc: VOC2007DetectionEvaluator,
    EvaluationType.bsd68: BSD68Evaluator,
    EvaluationType.bsd100: BSD100Evaluator,
    EvaluationType.widerface: WiderFaceEvaluator,
    EvaluationType.depth_estimation: DepthEstimationEvaluator,
    EvaluationType.instance_segmentation: InstanceSegEvaluator,
    EvaluationType.zeroshot_classification: CLIPEvaluator,
    EvaluationType.coco_pose: COCOPoseEvaluator,
    EvaluationType.obb: OBBEvaluator,
    EvaluationType.lfw: LFWEvaluator,
    EvaluationType.zeroshot_instance_segmentation: ZeroShotInstanceSegEvaluator,
    EvaluationType.hand_landmark: HandLandmarkEvaluator,
    EvaluationType.face_attribute: FaceAttributeEvaluator,
    EvaluationType.person_attribute: PersonAttributeEvaluator,
    EvaluationType.person_reid: PersonReIDEvaluator,
    EvaluationType.oxford_pet: SegentationEvaluator,
    EvaluationType.face_landmark: FaceLandmarkEvaluator,
    EvaluationType.lol: LOLEvaluator,
}
