from torchvision.transforms import Compose

from dx_modelzoo.enums import DatasetType, EvaluationType
from dx_modelzoo.models import ModelBase, ModelInfo
from dx_modelzoo.models.person_reid.osnet import reid_postprocessing
from dx_modelzoo.preprocessing.convertcolor import ConvertColor
from dx_modelzoo.preprocessing.div import Div
from dx_modelzoo.preprocessing.normalize import Normalize
from dx_modelzoo.preprocessing.resize import Resize
from dx_modelzoo.preprocessing.transpose import Transpose


class RepVGGA0_ReID(ModelBase):
    """RepVGG-A0 for person re-identification (512-dim).

    Input: [1, 3, 256, 128]
    Output: [1, 512] embedding vector
    """

    info = ModelInfo(
        name="RepVGGA0_ReID",
        dataset=DatasetType.market1501,
        evaluation=EvaluationType.person_reid,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(width=128, height=256),
                ConvertColor("BGR2RGB"),
                Div(255),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(width=128, height=256),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return reid_postprocessing
