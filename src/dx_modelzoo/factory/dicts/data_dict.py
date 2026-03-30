from dx_modelzoo.dataset.bsd68 import BSD68Dataset
from dx_modelzoo.dataset.bsd100 import BSD100Dataset
from dx_modelzoo.dataset.cbsd68 import CBSD68Dataset
from dx_modelzoo.dataset.celeba import CelebADataset
from dx_modelzoo.dataset.cityscapes import CityScapesDataset
from dx_modelzoo.dataset.coco import COCODataset
from dx_modelzoo.dataset.coco_pose import COCOPoseDataset
from dx_modelzoo.dataset.dotav1 import DOTAV1Dataset
from dx_modelzoo.dataset.face_landmark import FaceLandmarkDataset
from dx_modelzoo.dataset.hand_landmark import HandLandmarkDataset
from dx_modelzoo.dataset.imagenet import ImageNetDataset
from dx_modelzoo.dataset.lfw import LFWDataset
from dx_modelzoo.dataset.lol import LOLDataset
from dx_modelzoo.dataset.market1501 import Market1501Dataset
from dx_modelzoo.dataset.nyu import NYUDataset
from dx_modelzoo.dataset.oxford_pet import OxfordPetDataset
from dx_modelzoo.dataset.peta import PETADataset
from dx_modelzoo.dataset.voc import VOC2007Dection, VOCSegmentation
from dx_modelzoo.dataset.widerface import WiderFaceDataset
from dx_modelzoo.enums import DatasetType

DATASET_DICT = {
    DatasetType.imagenet: ImageNetDataset,
    DatasetType.coco: COCODataset,
    DatasetType.voc_seg: VOCSegmentation,
    DatasetType.voc_od: VOC2007Dection,
    DatasetType.bsd68: BSD68Dataset,
    DatasetType.cbsd68: CBSD68Dataset,
    DatasetType.bsd100: BSD100Dataset,
    DatasetType.city: CityScapesDataset,
    DatasetType.widerface: WiderFaceDataset,
    DatasetType.nyu: NYUDataset,
    DatasetType.coco_pose: COCOPoseDataset,
    DatasetType.dotav1: DOTAV1Dataset,
    DatasetType.lfw: LFWDataset,
    DatasetType.hand_landmark: HandLandmarkDataset,
    DatasetType.celeba: CelebADataset,
    DatasetType.peta: PETADataset,
    DatasetType.market1501: Market1501Dataset,
    DatasetType.oxford_pet: OxfordPetDataset,
    DatasetType.aflw2000_3d: FaceLandmarkDataset,
    DatasetType.lol: LOLDataset,
}
