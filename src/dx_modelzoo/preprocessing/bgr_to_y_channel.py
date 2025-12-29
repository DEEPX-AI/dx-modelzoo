import numpy as np
import torch


class BgrToYChannel:
    """Convert BGR image to Y channel (luminance) in YCbCr color space.
    
    Uses the same formula as ESPCN-PyTorch original implementation.
    """
    
    def __init__(self):
        pass
    
    def __call__(self, image) -> torch.Tensor:
        """Convert BGR image to Y channel exactly like ESPCN-PyTorch original.
        
        Args:
            image: BGR image in range [0, 255] with shape (H, W, 3), can be np.ndarray or torch.Tensor
            
        Returns:
            torch.Tensor: Y channel in range [0, 1] with shape (H, W)
        """
        # Convert to numpy if input is tensor
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        
        # Normalize to [0, 1] first (like original preprocess_one_image)
        image = image.astype(np.float32) / 255.0
        
        # BGR to YCbCr conversion (exactly like original bgr_to_ycbcr with only_use_y_channel=False)
        ycbcr_image = np.matmul(image, [[24.966, 112.0, -18.214], 
                                       [128.553, -74.203, -93.786], 
                                       [65.481, -37.797, 112.0]]) + [16, 128, 128]
        ycbcr_image /= 255.0
        ycbcr_image = ycbcr_image.astype(np.float32)
        
        # Extract Y channel (like cv2.split)
        y_channel = ycbcr_image[:, :, 0]
        
        # Convert to PyTorch tensor (2D, no extra channel dimension)
        return torch.from_numpy(y_channel)