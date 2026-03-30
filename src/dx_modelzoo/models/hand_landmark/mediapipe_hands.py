from typing import List

import numpy as np
import torch
from torchvision.transforms import Compose

from dx_modelzoo.enums import DatasetType, EvaluationType
from dx_modelzoo.models import ModelBase, ModelInfo
from dx_modelzoo.preprocessing.convertcolor import ConvertColor
from dx_modelzoo.preprocessing.div import Div
from dx_modelzoo.preprocessing.resize import Resize
from dx_modelzoo.preprocessing.transpose import Transpose


def hand_landmark_lite_postprocessing(outputs: List[np.ndarray]):
    """Postprocessing for HandLandmarkLite model.

    Model outputs:
        outputs[0]: [1, 63] - 21 landmarks (x, y, z), normalized to input size
        outputs[1]: [1, 1]  - handedness score
        outputs[2]: [1, 1]  - hand confidence/presence score
        outputs[3]: [1, 63] - world landmarks (3D metric space)

    Returns:
        Tuple of (keypoints, handedness, confidence, world_landmarks)
        - keypoints: [21, 3] tensor (x, y, z) normalized to [0, 1]
        - handedness: float
        - confidence: float
        - world_landmarks: [21, 3] tensor
    """
    landmarks = outputs[0]  # [1, 63]
    handedness = outputs[1]  # [1, 1]
    confidence = outputs[2]  # [1, 1]
    world_landmarks = outputs[3]  # [1, 63]

    if isinstance(landmarks, np.ndarray):
        landmarks = torch.from_numpy(landmarks)
    if isinstance(world_landmarks, np.ndarray):
        world_landmarks = torch.from_numpy(world_landmarks)

    # Reshape to [21, 3]
    keypoints = landmarks.squeeze(0).reshape(21, 3)
    world_kpts = world_landmarks.squeeze(0).reshape(21, 3)

    # Normalize x, y from pixel coords (224) to [0, 1]
    keypoints[:, 0] /= 224.0
    keypoints[:, 1] /= 224.0

    handedness_val = float(handedness.flatten()[0])
    confidence_val = float(confidence.flatten()[0])

    return keypoints, handedness_val, confidence_val, world_kpts


class MediaPipeHandsLite(ModelBase):
    """MediaPipe HandLandmarkLite model for 21-keypoint hand landmark estimation.

    Input: [1, 3, 224, 224] cropped hand image
    Output: 21 3D hand keypoints + handedness + confidence + world landmarks
    """

    info = ModelInfo(
        name="MediaPipeHandsLite",
        dataset=DatasetType.hand_landmark,
        evaluation=EvaluationType.hand_landmark,
    )

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="default", size=(224, 224)),
                ConvertColor("BGR2RGB"),
                Div(255),
                Transpose([2, 0, 1]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="default", size=(224, 224)),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return hand_landmark_lite_postprocessing
