import numpy as np
import torch
from torchvision.transforms import Compose

from dx_modelzoo.enums import DatasetType, EvaluationType, SessionType
from dx_modelzoo.models import ModelBase, ModelInfo
from dx_modelzoo.models.face_detection import non_max_suppression_for_scrfd, scale_coords_for_yolov5_face
from dx_modelzoo.preprocessing.convertcolor import ConvertColor
from dx_modelzoo.preprocessing.normalize import Normalize
from dx_modelzoo.preprocessing.resize import Resize
from dx_modelzoo.preprocessing.transpose import Transpose


def postprocess_scrfd(net_outs, model_input_size, original_img_shape, session, conf_thres=0.02, iou_thres=0.45):
    if session.type == SessionType.dxruntime:
        model_input_size = [model_input_size[0], model_input_size[3], model_input_size[1], model_input_size[2]]
    w_model, h_model = model_input_size[3], model_input_size[2]
    model_input_size = (h_model, w_model)

    strides_config = [8, 16, 32]
    all_grids, all_strides = [], []
    for stride in strides_config:
        h, w = h_model // stride, w_model // stride
        if h <= 0 or w <= 0:
            continue

        grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        grid = torch.stack((grid_x, grid_y), 2).view(-1, 2)
        all_grids.append(grid)
        all_strides.append(torch.full((grid.shape[0], 1), stride))

    if not all_grids:
        return []

    grids_base = torch.cat(all_grids, dim=0)
    strides_base = torch.cat(all_strides, dim=0)

    grids = grids_base.repeat_interleave(2, dim=0).float()
    strides_tensor = strides_base.repeat_interleave(2, dim=0).float()
    conf_thres = 0.02
    iou_thres = 0.45

    conf = np.concatenate(
        sorted([n_out for n_out in net_outs if n_out.shape[-1] == 1], key=lambda x: x.shape[1], reverse=True), axis=1
    )
    box = np.concatenate(
        sorted([n_out for n_out in net_outs if n_out.shape[-1] == 4], key=lambda x: x.shape[1], reverse=True), axis=1
    )
    conf = torch.from_numpy(conf)
    box = torch.from_numpy(box)

    box = torch.cat([grids - box[:, :, 0:2], grids + box[:, :, 2:4]], dim=2) * strides_tensor
    pred = torch.cat([box, conf], dim=2)

    pred = non_max_suppression_for_scrfd(pred, conf_thres=conf_thres, iou_thres=iou_thres)[0]
    if pred is not None:
        box, conf = pred[:, :4], pred[:, 4:]
        box = scale_coords_for_yolov5_face(model_input_size, box, original_img_shape[:2])
        box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
        box = box.round()
        pred = torch.cat([box, conf], dim=1)
    return pred


class SCRFD10G(ModelBase):
    info = ModelInfo(name="SCRFD10G", dataset=DatasetType.widerface, evaluation=EvaluationType.widerface)

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=640, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor("BGR2RGB"),
                Transpose([2, 0, 1]),
                Normalize(mean=[127.5, 127.5, 127.5], std=[128.0, 128.0, 128.0]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=640, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return postprocess_scrfd


class SCRFD2_5G(ModelBase):
    info = ModelInfo(name="SCRFD2_5G", dataset=DatasetType.widerface, evaluation=EvaluationType.widerface)

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=640, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor("BGR2RGB"),
                Transpose([2, 0, 1]),
                Normalize(mean=[127.5, 127.5, 127.5], std=[128.0, 128.0, 128.0]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=640, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return postprocess_scrfd


class SCRFD500M(ModelBase):
    info = ModelInfo(name="SCRFD500M", dataset=DatasetType.widerface, evaluation=EvaluationType.widerface)

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=640, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor("BGR2RGB"),
                Transpose([2, 0, 1]),
                Normalize(mean=[127.5, 127.5, 127.5], std=[128.0, 128.0, 128.0]),
            ]
        )

    def npu_preprocessing(self):
        return Compose(
            [
                Resize(mode="pad", size=640, pad_location="edge", pad_value=[114, 114, 114]),
                ConvertColor("BGR2RGB"),
            ]
        )

    def postprocessing(self):
        return postprocess_scrfd
