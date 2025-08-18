from dx_modelzoo.enums import DatasetType, EvaluationType
from dx_modelzoo.models import ModelBase, ModelInfo
from dx_modelzoo.models.object_dection.nms import ssd_nms
# from dx_modelzoo.models.object_dection.nms import ssd_nms_wrapper


class SSDMV1(ModelBase):
    info = ModelInfo(name="SSDMV1", dataset=DatasetType.voc_od, evaluation=EvaluationType.voc)

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return [
            {"convertColor": {"form": "BGR2RGB"}},
            {"resize": {"mode": "default", "width": 300, "height": 300}},
            {"normalize": {"mean": [127, 127, 127], "std": [128.0, 128.0, 128.0]}},
            {"transpose": {"axis": [2, 0, 1]}},
            {"expandDim": {"axis": 0}},
        ]

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
        return [
            {"convertColor": {"form": "BGR2RGB"}},
            {"resize": {"mode": "default", "width": 300, "height": 300}},
            {"normalize": {"mean": [127, 127, 127], "std": [128.0, 128.0, 128.0]}},
            {"transpose": {"axis": [2, 0, 1]}},
            {"expandDim": {"axis": 0}},
        ]

    def postprocessing(self):
        return ssd_nms
        # # Note: Temporary workaround for the mismatch in output tensor order between the original ONNX model and DXNN.
        # #       The _wrapper function will be removed once the issue is properly fixed.
        # return ssd_nms_wrapper
