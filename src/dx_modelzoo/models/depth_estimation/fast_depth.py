import numpy as np
from PIL import Image
from torchvision.transforms import Compose

from dx_modelzoo.enums import DatasetType, EvaluationType
from dx_modelzoo.models import ModelBase, ModelInfo
from dx_modelzoo.preprocessing.centercrop import CenterCrop
from dx_modelzoo.preprocessing.div import Div
from dx_modelzoo.preprocessing.expanddim import ExpandDim
from dx_modelzoo.preprocessing.transpose import Transpose


def fastdepth_postprocessing(inputs):
    return inputs[0]


class Resize:
    """
    A class for resizing images, similar to the functionality of scipy.misc.imresize.

    This class uses the Pillow library to resize images. The `size` parameter can be an
    integer (percentage), a float (ratio), or a tuple of (width, height).

    Args:
        size (int, float, or tuple):
            - int: The percentage of the current size.
            - float: The ratio of the current size.
            - tuple: The output image size in the format (width, height).
        interpolation (str, optional):
            The interpolation method to use for resizing.
            Options are: 'nearest', 'bilinear', 'bicubic', 'lanczos'.
            Defaults to 'bilinear'.
    """

    _interpolation_map = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }

    def __init__(self, size, interpolation="bilinear"):
        if not isinstance(size, (int, float, tuple)):
            raise TypeError("size must be an integer, float, or tuple.")
        if interpolation not in self._interpolation_map:
            raise ValueError(f"Unsupported interpolation type: {interpolation}")

        self.size = size
        self.interpolation = self._interpolation_map[interpolation]

    def __call__(self, img_array):
        """
        Resizes the given NumPy array image.

        Args:
            img_array (np.ndarray): The image array to be resized.

        Returns:
            np.ndarray: The resized image array.
        """
        img = Image.fromarray(img_array)
        original_size = img.size  # (width, height)

        if isinstance(self.size, int):
            new_width = original_size[0] * self.size // 100
            new_height = original_size[1] * self.size // 100
            new_size = (new_width, new_height)
        elif isinstance(self.size, float):
            new_width = int(original_size[0] * self.size)
            new_height = int(original_size[1] * self.size)
            new_size = (new_width, new_height)
        else:  # tuple
            new_size = self.size

        resized_img = img.resize(new_size, self.interpolation)
        return np.array(resized_img)


class FastDepth(ModelBase):
    info = ModelInfo(name="FastDepth", dataset=DatasetType.nyu, evaluation=EvaluationType.depth_estimation)

    def __init__(self, evaluator):
        super().__init__(evaluator)

        self.evaluator.dataset.depth_preprocessing = self.depth_preprocessing()

    def depth_preprocessing(self):
        return Compose([Resize(250.0 / 480), CenterCrop(228, 304), Resize((224, 224)), ExpandDim(0)])

    def preprocessing(self):
        return Compose([Resize(250.0 / 480), CenterCrop(228, 304), Resize((224, 224)), Div(255), Transpose([2, 0, 1])])

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(250.0 / 480),
                CenterCrop(228, 304),
                Resize((224, 224)),
            ]
        )

    def postprocessing(self):
        return fastdepth_postprocessing
