import numpy as np
import torch


class BgrToYChannelUint8:
    """Convert BGR image to Y channel (luminance) in YCbCr color space.

    Returns Y channel in [0, 255] uint8 range for DXNN models.
    """

    def __init__(self):
        pass

    def __call__(self, image) -> torch.Tensor:
        """Convert BGR image to Y channel in [0, 255] range.

        Args:
            image: BGR image in range [0, 255] with shape (H, W, 3), can be np.ndarray or torch.Tensor

        Returns:
            torch.Tensor: Y channel in range [0, 255] uint8 with shape (H, W)
        """
        # Convert to numpy if input is tensor
        if isinstance(image, torch.Tensor):
            image = image.numpy()

        # Normalize to [0, 1] first for calculation
        image = image.astype(np.float32) / 255.0

        # BGR to YCbCr conversion
        ycbcr_image = np.matmul(
            image, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]
        ) + [16, 128, 128]

        # Extract Y channel and keep in [0, 255] range (round and clip)
        y_channel = ycbcr_image[:, :, 0]
        y_channel = np.clip(y_channel, 0, 255).astype(np.uint8)

        return torch.from_numpy(y_channel)
