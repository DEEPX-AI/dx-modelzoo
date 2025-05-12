import numpy as np

from dx_modelzoo.dataset.imagenet import ImageNetDataset
from dx_modelzoo.preprocessing.div import Div


def test_imagenet(datadir):
    imagenetdataset = ImageNetDataset(datadir)
    imagenetdataset.preprocessing = Div(1)

    assert len(imagenetdataset) == 9
    assert imagenetdataset.class_map["class_a"] == 0
    assert imagenetdataset.class_map["class_b"] == 1
    assert imagenetdataset.class_map["class_c"] == 2

    for i in range(3):
        assert imagenetdataset[i][1] == 0
    for i in range(3):
        assert imagenetdataset[i + 3][1] == 1
    for i in range(3):
        assert imagenetdataset[i + 6][1] == 2

    for i in range(9):
        assert isinstance(imagenetdataset[i][0], np.ndarray)
