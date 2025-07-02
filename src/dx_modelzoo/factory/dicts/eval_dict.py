from dx_modelzoo.enums import EvaluationType
from dx_modelzoo.evaluator.bsd68_evaluator import BSD68Evaluator
from dx_modelzoo.evaluator.coco_evaluator import COCOEvaluator
from dx_modelzoo.evaluator.ic_evaluator import ICEvaluator
from dx_modelzoo.evaluator.segmentation_evaluator import SegentationEvaluator
from dx_modelzoo.evaluator.voc_evaluator import VOC2007DetectionEvaluator
from dx_modelzoo.evaluator.widerface_evaluator import WiderFaceEvaluator

EVAL_DICT = {
    EvaluationType.image_classification: ICEvaluator,
    EvaluationType.coco: COCOEvaluator,
    EvaluationType.segmentation: SegentationEvaluator,
    EvaluationType.voc: VOC2007DetectionEvaluator,
    EvaluationType.bsd68: BSD68Evaluator,
    EvaluationType.widerface: WiderFaceEvaluator,
}
