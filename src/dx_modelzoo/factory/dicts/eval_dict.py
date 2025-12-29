from dx_modelzoo.enums import EvaluationType
from dx_modelzoo.evaluator.bsd68_evaluator import BSD68Evaluator
from dx_modelzoo.evaluator.bsd100_evaluator import BSD100Evaluator
from dx_modelzoo.evaluator.clip_evaluator import CLIPEvaluator
from dx_modelzoo.evaluator.coco_evaluator import COCOEvaluator
from dx_modelzoo.evaluator.depth_estimation_evaluator import DepthEstimationEvaluator
from dx_modelzoo.evaluator.ic_evaluator import ICEvaluator
from dx_modelzoo.evaluator.instance_seg_evaluator import InstanceSegEvaluator
from dx_modelzoo.evaluator.segmentation_evaluator import SegentationEvaluator
from dx_modelzoo.evaluator.voc_evaluator import VOC2007DetectionEvaluator
from dx_modelzoo.evaluator.widerface_evaluator import WiderFaceEvaluator
from dx_modelzoo.evaluator.coco_pose_evaluator import COCOPoseEvaluator

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
}
