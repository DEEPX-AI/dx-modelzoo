from typing import Callable
import numpy as np

from torchvision.transforms import Compose

from dx_modelzoo.enums import DatasetType, EvaluationType
from dx_modelzoo.evaluator.bsd100_evaluator import BSD100Evaluator
from dx_modelzoo.models import ModelBase, ModelInfo
from dx_modelzoo.preprocessing.centercrop import CenterCrop
from dx_modelzoo.preprocessing.transpose import Transpose
from dx_modelzoo.preprocessing.bgr_to_y_channel import BgrToYChannel
from dx_modelzoo.preprocessing.expanddim import ExpandDim


def super_resolution_postprocessing(upscale_factor: int) -> Callable:
    """Post-processing for ESPCN super resolution models.
    
    ESPCN uses sub-pixel convolution, so the output needs depth-to-space conversion.
    
    Args:
        upscale_factor (int): The upscaling factor (2, 3, 4, or 8)
    
    Returns:
        Callable: Function to convert sub-pixel convolution output to proper image format
    """
    
    def postprocess(x):
        if isinstance(x, list) and len(x) > 0:
            x = x[0]
            
        batch_size, channels_squared, h, w = x.shape
        
        out_channels = 1
        
        x = x.reshape(batch_size, out_channels, upscale_factor, upscale_factor, h, w)
        x = x.transpose(0, 1, 4, 2, 5, 3)
        x = x.reshape(batch_size, out_channels, h * upscale_factor, w * upscale_factor)
        
        x = np.clip(x, 0.0, 1.0)
        
        return x
    return postprocess


class ESPCNBase(ModelBase):
    """Base class for ESPCN models with different upscale factors."""
    
    def __init__(self, evaluator: BSD100Evaluator, upscale_factor: int) -> None:
        self.upscale_factor = upscale_factor
        super().__init__(evaluator)
        
        if hasattr(evaluator.dataset, 'set_lr_preprocessing'):
            evaluator.dataset.set_lr_preprocessing(self.preprocessing())
        if hasattr(evaluator.dataset, 'set_hr_preprocessing'):
            evaluator.dataset.set_hr_preprocessing(self.hr_preprocessing())

    def preprocessing(self) -> Compose:
        """Preprocessing for ESPCN LR input.
        
        DXNN model expects 17x17 fixed input size.
        """
        return Compose([
            CenterCrop(17, 17),
            BgrToYChannel(),
            ExpandDim(axis=0), 
        ])

    def npu_preprocessing(self) -> Compose:
        """NPU preprocessing for ESPCN LR input."""
        return Compose([
            CenterCrop(17, 17),
            BgrToYChannel(),
            Transpose(axis=None),
        ])

    def hr_preprocessing(self) -> Compose:
        """Preprocessing for HR ground truth images.
        
        HR images need to match the expected output size (17 * upscale_factor).
        """
        hr_size = 17 * self.upscale_factor
        return Compose([
            CenterCrop(hr_size, hr_size),
            BgrToYChannel(),
            ExpandDim(axis=0), 
        ])
    
    def postprocessing(self) -> Callable:
        return super_resolution_postprocessing(self.upscale_factor)


class ESPCN_x2(ESPCNBase):
    """ESPCN model with 2x upscale factor."""
    
    info = ModelInfo(
        name="ESPCN_x2",
        dataset=DatasetType.bsd100,
        evaluation=EvaluationType.bsd68,
        raw_performance="36.64 0.9542",
        q_lite_performance="36.50 0.9535",
        q_pro_performance="36.45 0.9530",
        q_master_performance="36.40 0.9525"
    )

    def __init__(self, evaluator: BSD100Evaluator) -> None:
        super().__init__(evaluator, upscale_factor=2)


class ESPCN_x3(ESPCNBase):
    """ESPCN model with 3x upscale factor."""
    
    info = ModelInfo(
        name="ESPCN_x3", 
        dataset=DatasetType.bsd100,
        evaluation=EvaluationType.bsd68,
        raw_performance="32.55 0.9088",  # PSNR SSIM
        q_lite_performance="32.45 0.9080",
        q_pro_performance="32.40 0.9075", 
        q_master_performance="32.35 0.9070"
    )

    def __init__(self, evaluator: BSD100Evaluator) -> None:
        super().__init__(evaluator, upscale_factor=3)


class ESPCN_x4(ESPCNBase):
    """ESPCN model with 4x upscale factor."""
    
    info = ModelInfo(
        name="ESPCN_x4",
        dataset=DatasetType.bsd100, 
        evaluation=EvaluationType.bsd68,
        raw_performance="30.26 0.8732",  # PSNR SSIM
        q_lite_performance="30.15 0.8720",
        q_pro_performance="30.10 0.8715",
        q_master_performance="30.05 0.8710"
    )

    def __init__(self, evaluator: BSD100Evaluator) -> None:
        super().__init__(evaluator, upscale_factor=4)


class ESPCN_x8(ESPCNBase):
    """ESPCN model with 8x upscale factor."""
    
    info = ModelInfo(
        name="ESPCN_x8",
        dataset=DatasetType.bsd100,
        evaluation=EvaluationType.bsd68,
        raw_performance="26.80 0.7800",
        q_lite_performance="26.70 0.7790",
        q_pro_performance="26.65 0.7785",
        q_master_performance="26.60 0.7780"
    )

    def __init__(self, evaluator: BSD100Evaluator) -> None:
        super().__init__(evaluator, upscale_factor=8)