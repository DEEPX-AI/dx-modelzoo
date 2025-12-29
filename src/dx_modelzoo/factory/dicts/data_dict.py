from dx_modelzoo.dataset.bsd68 import BSD68Dataset
from dx_modelzoo.dataset.bsd100 import BSD100Dataset
from dx_modelzoo.dataset.cityspaces import CitySpaceDataset
from dx_modelzoo.dataset.coco import COCODataset
from dx_modelzoo.dataset.imagenet import ImageNetDataset
from dx_modelzoo.dataset.nyu import NYUDataset
from dx_modelzoo.dataset.voc import VOC2007Dection, VOCSegmentation
from dx_modelzoo.dataset.widerface import WiderFaceDataset
from dx_modelzoo.dataset.coco_pose import COCOPoseDataset
from dx_modelzoo.enums import DatasetType

DATASET_DICT = {
    DatasetType.imagenet: ImageNetDataset,
    DatasetType.coco: COCODataset,
    DatasetType.voc_seg: VOCSegmentation,
    DatasetType.voc_od: VOC2007Dection,
    DatasetType.bsd68: BSD68Dataset,
    DatasetType.bsd100: BSD100Dataset,
    DatasetType.city: CitySpaceDataset,
    DatasetType.widerface: WiderFaceDataset,
    DatasetType.nyu: NYUDataset,
    DatasetType.coco_pose: COCOPoseDataset,
}
