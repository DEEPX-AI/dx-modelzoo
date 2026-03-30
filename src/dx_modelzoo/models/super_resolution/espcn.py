from typing import Callable

import numpy as np
from torchvision.transforms import Compose

from dx_modelzoo.enums import DatasetType, EvaluationType, SessionType
from dx_modelzoo.evaluator.bsd100_evaluator import BSD100Evaluator
from dx_modelzoo.models import ModelBase, ModelInfo
from dx_modelzoo.preprocessing.bgr_to_y_channel import BgrToYChannel
from dx_modelzoo.preprocessing.bgr_to_y_channel_uint8 import BgrToYChannelUint8
from dx_modelzoo.preprocessing.centercrop import CenterCrop
from dx_modelzoo.preprocessing.expanddim import ExpandDim


def super_resolution_postprocessing(upscale_factor: int) -> Callable:
    """Post-processing for ESPCN super resolution models.

    ESPCN uses sub-pixel convolution, so the output needs depth-to-space conversion.
    Handles both ONNX output (sub-pixel format) and DXNN output (DepthToSpace already applied).

    Args:
        upscale_factor (int): The upscaling factor (2, 3, 4, or 8)

    Returns:
        Callable: Function to convert sub-pixel convolution output to proper image format
    """

    def postprocess(x):
        if isinstance(x, list) and len(x) > 0:
            x = x[0]

        # Convert to float if needed
        x = x.astype(np.float32)

        batch_size, channels, h, w = x.shape

        # Check if DepthToSpace was already applied by DXNN
        # ONNX output: (batch, upscale_factor^2, h, w) e.g., (1, 4, 17, 17) for x2
        # DXNN output: (batch, 1, h*upscale_factor, w*upscale_factor) e.g., (1, 1, 34, 34) for x2
        expected_channels = upscale_factor * upscale_factor

        if channels == expected_channels:
            # ONNX format: needs depth-to-space conversion
            out_channels = 1
            x = x.reshape(batch_size, out_channels, upscale_factor, upscale_factor, h, w)
            x = x.transpose(0, 1, 4, 2, 5, 3)
            x = x.reshape(batch_size, out_channels, h * upscale_factor, w * upscale_factor)
        else:
            # DXNN format - DepthToSpace already applied
            # DXNN output is usually in 0-255 range, normalize to 0-1
            if x.max() > 1.0:
                x = x / 255.0

        x = np.clip(x, 0.0, 1.0)

        return x

    return postprocess


class ESPCNBase(ModelBase):
    """Base class for ESPCN models with different upscale factors."""

    def __init__(self, evaluator: BSD100Evaluator, upscale_factor: int) -> None:
        self.upscale_factor = upscale_factor
        super().__init__(evaluator)
        self.evaluator.upscale_factor = upscale_factor

        self.lr_preprocessing = (
            self.preprocessing() if evaluator.session.type == SessionType.onnxruntime else self.npu_preprocessing()
        )

        if hasattr(evaluator.dataset, "set_lr_preprocessing"):
            evaluator.dataset.set_lr_preprocessing(self.lr_preprocessing)
        if hasattr(evaluator.dataset, "set_hr_preprocessing"):
            evaluator.dataset.set_hr_preprocessing(self.hr_preprocessing())

    def preprocessing(self) -> Compose:
        """Preprocessing for ESPCN LR input.

        DXNN model expects 17x17 fixed input size.
        """
        return Compose(
            [
                CenterCrop(17, 17),
                BgrToYChannel(),
                ExpandDim(axis=0),
            ]
        )

    def npu_preprocessing(self) -> Compose:
        """NPU preprocessing for ESPCN LR input.

        DXNN expects input in [0, 255] range (uint8) with NHWC format (1, H, W, 1).
        """
        return Compose(
            [
                CenterCrop(17, 17),
                BgrToYChannelUint8(),
                ExpandDim(axis=0),  # (H, W) -> (1, H, W)
                ExpandDim(axis=-1),  # (1, H, W) -> (1, H, W, 1)
            ]
        )

    def hr_preprocessing(self) -> Compose:
        """Preprocessing for HR ground truth images.

        HR images need to match the expected output size (17 * upscale_factor).
        """
        hr_size = 17 * self.upscale_factor
        return Compose(
            [
                CenterCrop(hr_size, hr_size),
                BgrToYChannel(),
                ExpandDim(axis=0),
            ]
        )

    def postprocessing(self) -> Callable:
        return super_resolution_postprocessing(self.upscale_factor)


class ESPCN_x2(ESPCNBase):
    """ESPCN model with 2x upscale factor."""

    info = ModelInfo(
        name="ESPCN_x2",
        dataset=DatasetType.bsd100,
        evaluation=EvaluationType.bsd100,
        raw_performance="36.64 0.9542",
        q_lite_performance="36.50 0.9535",
        q_pro_performance="36.45 0.9530",
        q_master_performance="36.40 0.9525",
    )

    def __init__(self, evaluator: BSD100Evaluator) -> None:
        super().__init__(evaluator, upscale_factor=2)


class ESPCN_x3(ESPCNBase):
    """ESPCN model with 3x upscale factor."""

    info = ModelInfo(
        name="ESPCN_x3",
        dataset=DatasetType.bsd100,
        evaluation=EvaluationType.bsd100,
        raw_performance="32.55 0.9088",  # PSNR SSIM
        q_lite_performance="32.45 0.9080",
        q_pro_performance="32.40 0.9075",
        q_master_performance="32.35 0.9070",
    )

    def __init__(self, evaluator: BSD100Evaluator) -> None:
        super().__init__(evaluator, upscale_factor=3)


class ESPCN_x4(ESPCNBase):
    """ESPCN model with 4x upscale factor."""

    info = ModelInfo(
        name="ESPCN_x4",
        dataset=DatasetType.bsd100,
        evaluation=EvaluationType.bsd100,
        raw_performance="30.26 0.8732",  # PSNR SSIM
        q_lite_performance="30.15 0.8720",
        q_pro_performance="30.10 0.8715",
        q_master_performance="30.05 0.8710",
    )

    def __init__(self, evaluator: BSD100Evaluator) -> None:
        super().__init__(evaluator, upscale_factor=4)


class ESPCN_x8(ESPCNBase):
    """ESPCN model with 8x upscale factor."""

    info = ModelInfo(
        name="ESPCN_x8",
        dataset=DatasetType.bsd100,
        evaluation=EvaluationType.bsd100,
        raw_performance="26.80 0.7800",
        q_lite_performance="26.70 0.7790",
        q_pro_performance="26.65 0.7785",
        q_master_performance="26.60 0.7780",
    )

    def __init__(self, evaluator: BSD100Evaluator) -> None:
        super().__init__(evaluator, upscale_factor=8)
