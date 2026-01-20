from torchvision.transforms import Compose

from dx_modelzoo.enums import DatasetType, EvaluationType
from dx_modelzoo.models import ModelBase, ModelInfo

# from dx_modelzoo.models.object_detection.nms import ssd_nms_wrapper
from dx_modelzoo.models.object_detection.nms import ssd_nms, ssd_nms_wrapper
from dx_modelzoo.preprocessing.convertcolor import ConvertColor
from dx_modelzoo.preprocessing.normalize import Normalize
from dx_modelzoo.preprocessing.resize import Resize
from dx_modelzoo.preprocessing.transpose import Transpose


class SSDMV1(ModelBase):
    info = ModelInfo(name="SSDMV1", dataset=DatasetType.voc_od, evaluation=EvaluationType.voc)

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                ConvertColor("BGR2RGB"),
                Resize(width=300, height=300),
                Normalize([127, 127, 127], [128.0, 128.0, 128.0]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                ConvertColor("BGR2RGB"),
                Resize(width=300, height=300),
            ]
        )

    def postprocessing(self):
        return ssd_nms
        # # Note: Temporary workaround for the mismatch in output tensor order between the original ONNX model and DXNN.
        # #       The _wrapper function will be removed once the issue is properly fixed.
        # return ssd_nms_wrapper



class SSDMV2Lite(ModelBase):
    info = ModelInfo(name="SSDMV2Lite", dataset=DatasetType.voc_od, evaluation=EvaluationType.voc)

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                ConvertColor("BGR2RGB"),
                Resize(width=300, height=300),
                Normalize([127, 127, 127], [128.0, 128.0, 128.0]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                ConvertColor("BGR2RGB"),
                Resize(width=300, height=300),
            ]
        )

    def postprocessing(self):
        return ssd_nms
        # # Note: Temporary workaround for the mismatch in output tensor order between the original ONNX model and DXNN.
        # #       The _wrapper function will be removed once the issue is properly fixed.
        # return ssd_nms_wrapper
