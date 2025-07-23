from typing import List, Tuple

import torch

from dx_modelzoo.enums import DatasetType, EvaluationType, SessionType
from dx_modelzoo.models import ModelBase, ModelInfo
from dx_modelzoo.models.face_detection import non_max_suppression_for_yolv5_face, non_max_suppression_for_yolv7_face
from dx_modelzoo.utils.detection import clip_boxes


def get_ratios(inp_image_shape, origin_img_shape):
    return min(inp_image_shape[2] / origin_img_shape[0], inp_image_shape[3] / origin_img_shape[1])


def get_pad_size(inp_image_shape, origin_img_shape, ratios):
    return (inp_image_shape[2] - origin_img_shape[1] * ratios) / 2, (
        inp_image_shape[3] - origin_img_shape[0] * ratios
    ) / 2  # wh padding


def scale_boxes(
    boxes: torch.Tensor, origin_shape: List[torch.Tensor], ratios: torch.Tensor, pads: Tuple[torch.Tensor, torch.Tensor]
):
    boxes[:, [0, 2]] -= pads[0]  # x padding
    boxes[:, [1, 3]] -= pads[1]  # y padding
    boxes[:, :4] /= ratios
    return clip_boxes(boxes, origin_shape)


def make_xywh(x):
    y = x.clone()
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def yolov5_face_postprocessing_wrapper(outputs, inp_shape, origin_shape, session):
    if session.type == SessionType.onnxruntime:
        pass
    elif session.type == SessionType.dxruntime:
        output_tensors_info = session.inference_engine.get_output_tensors_info()
        idx = find_non_onnx_slice_index(output_tensors_info)
        outputs = [outputs[idx]]
    else:
        raise Exception(f"Invalid SeessionType: {session.type}")
    
    return yolov5_face_postprocessing(outputs, inp_shape, origin_shape)


def yolov5_face_postprocessing(outputs, inp_shape, origin_shape):
    h, w, _ = origin_shape

    outputs = torch.from_numpy(outputs[0])
    outputs = non_max_suppression_for_yolv5_face(outputs, 0.02, 0.5)
    cloned = outputs.clone()
    ratios = get_ratios(inp_shape, origin_shape)
    pads = get_pad_size(inp_shape, origin_shape, ratios)
    scaled_boxes = scale_boxes(cloned[..., :4], origin_shape, ratios, pads).round()
    norm = torch.tensor(origin_shape)[[1, 0, 1, 0]]  # normalization gain whwh

    final_boxes = []
    for box_idx in range(scaled_boxes.size(0)):
        xywh = (make_xywh(scaled_boxes[box_idx].view(1, 4)) / norm).view(-1)
        xywh = xywh.data.cpu().numpy()
        conf = outputs[box_idx, 4].cpu().numpy()
        x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
        y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
        x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
        y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
        final_boxes.append([x1, y1, x2 - x1, y2 - y1, conf])
    return final_boxes


class YOLOv5s_Face(ModelBase):
    info = ModelInfo(name="YOLOv5n0_5_Face", dataset=DatasetType.widerface, evaluation=EvaluationType.widerface)

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return [
            {"resize": {"mode": "pad", "size": 640, "pad_location": "edge", "pad_value": [114, 114, 114]}},
            {"convertColor": {"form": "BGR2RGB"}},
            {"transpose": {"axis": [2, 0, 1]}},
            {"div": {"x": 255}},
            {"expandDim": {"axis": 0}},
        ]

    def postprocessing(self):
        return yolov5_face_postprocessing_wrapper


class YOLOv5m_Face(ModelBase):
    info = ModelInfo(name="YOLOv5m_Face", dataset=DatasetType.widerface, evaluation=EvaluationType.widerface)

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return [
            {"resize": {"mode": "pad", "size": 640, "pad_location": "edge", "pad_value": [114, 114, 114]}},
            {"convertColor": {"form": "BGR2RGB"}},
            {"transpose": {"axis": [2, 0, 1]}},
            {"div": {"x": 255.0}},
            {"expandDim": {"axis": 0}},
        ]

    def postprocessing(self):
        return yolov5_face_postprocessing_wrapper

def find_non_onnx_slice_index(data):
    indices = [i for i, item in enumerate(data) if 'onnx::Slice' not in item['name']]
    if indices.__len__() != 1:
        raise Exception(f"Expected exactly one output tensor, but found a different number, num of output tensor: {indices.__len__()}")
    else:
        return indices[0]

# Note: Temporary workaround for the mismatch in output tensor order between the original ONNX model and DXNN.
#       The _wrapper function will be removed once the issue is properly fixed.
def yolov7_face_postprocessing_wrapper(outputs, inp_shape, origin_shape, session):
    if session.type == SessionType.onnxruntime:
        pass
    elif session.type == SessionType.dxruntime:
        output_tensors_info = session.inference_engine.get_output_tensors_info()
        idx = find_non_onnx_slice_index(output_tensors_info)
        outputs = [outputs[idx]]
    else:
        raise Exception(f"Invalid SeessionType: {session.type}")
    
    return yolov7_face_postprocessing(outputs, inp_shape, origin_shape)

def yolov7_face_postprocessing(outputs, inp_shape, origin_shape):
    
    outputs = torch.from_numpy(outputs[0])
    outputs = non_max_suppression_for_yolv7_face(outputs, 0.01, 0.5)

    cloned = outputs.clone()
    ratios = get_ratios(inp_shape, origin_shape)
    pads = get_pad_size(inp_shape, origin_shape, ratios)
    scaled_boxes = scale_boxes(cloned[..., :4], origin_shape, ratios, pads)

    final_boxes = []
    for box_idx in range(scaled_boxes.size(0)):
        x1 = int(scaled_boxes[box_idx, 0] + 0.5)
        y1 = int(scaled_boxes[box_idx, 1] + 0.5)
        x2 = int(scaled_boxes[box_idx, 2] + 0.5)
        y2 = int(scaled_boxes[box_idx, 3] + 0.5)
        final_boxes.append([x1, y1, x2 - x1, y2 - y1, outputs[box_idx, 4]])
    return final_boxes


class YOLOv7_Face(ModelBase):
    info = ModelInfo(name="YOLOv7_Face", dataset=DatasetType.widerface, evaluation=EvaluationType.widerface)

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return [
            {"resize": {"mode": "pad", "size": 640, "pad_location": "edge", "pad_value": [114, 114, 114]}},
            {"convertColor": {"form": "BGR2RGB"}},
            {"transpose": {"axis": [2, 0, 1]}},
            {"div": {"x": 255.0}},
            {"expandDim": {"axis": 0}},
        ]

    def postprocessing(self):
        # Note: Temporary workaround for the mismatch in output tensor order between the original ONNX model and DXNN.
        #       The _wrapper function will be removed once the issue is properly fixed.
        # return yolov7_face_postprocessing
        return yolov7_face_postprocessing_wrapper


class YOLOv7s_Face(ModelBase):
    info = ModelInfo(name="YOLOv7s_Face", dataset=DatasetType.widerface, evaluation=EvaluationType.widerface)

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return [
            {"resize": {"mode": "pad", "size": 640, "pad_location": "edge", "pad_value": [114, 114, 114]}},
            {"convertColor": {"form": "BGR2RGB"}},
            {"transpose": {"axis": [2, 0, 1]}},
            {"div": {"x": 255.0}},
            {"expandDim": {"axis": 0}},
        ]

    def postprocessing(self):
        # Note: Temporary workaround for the mismatch in output tensor order between the original ONNX model and DXNN.
        #       The _wrapper function will be removed once the issue is properly fixed.
        # return yolov7_face_postprocessing
        return yolov7_face_postprocessing_wrapper


class YOLOv7_TTA_Face(ModelBase):
    info = ModelInfo(name="YOLOv7_TTA_Face", dataset=DatasetType.widerface, evaluation=EvaluationType.widerface)

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return [
            {"resize": {"mode": "pad", "size": 640, "pad_location": "edge", "pad_value": [114, 114, 114]}},
            {"convertColor": {"form": "BGR2RGB"}},
            {"transpose": {"axis": [2, 0, 1]}},
            {"div": {"x": 255.0}},
            {"expandDim": {"axis": 0}},
        ]

    def postprocessing(self):
        # Note: Temporary workaround for the mismatch in output tensor order between the original ONNX model and DXNN.
        #       The _wrapper function will be removed once the issue is properly fixed.
        # return yolov7_face_postprocessing
        return yolov7_face_postprocessing_wrapper


class YOLOv7_W6_Face(ModelBase):
    info = ModelInfo(name="YOLOv7_W6_Face", dataset=DatasetType.widerface, evaluation=EvaluationType.widerface)

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return [
            {"resize": {"mode": "pad", "size": 960, "pad_location": "edge", "pad_value": [114, 114, 114]}},
            {"convertColor": {"form": "BGR2RGB"}},
            {"transpose": {"axis": [2, 0, 1]}},
            {"div": {"x": 255.0}},
            {"expandDim": {"axis": 0}},
        ]

    def postprocessing(self):
        # Note: Temporary workaround for the mismatch in output tensor order between the original ONNX model and DXNN.
        #       The _wrapper function will be removed once the issue is properly fixed.
        # return yolov7_face_postprocessing
        return yolov7_face_postprocessing_wrapper


class YOLOv7_W6_TTA_Face(ModelBase):
    info = ModelInfo(name="YOLOv7_W6_TTA_Face", dataset=DatasetType.widerface, evaluation=EvaluationType.widerface)

    def __init__(self, evaluator):
        super().__init__(evaluator)

    def preprocessing(self):
        return [
            {"resize": {"mode": "pad", "size": 1280, "pad_location": "edge", "pad_value": [114, 114, 114]}},
            {"convertColor": {"form": "BGR2RGB"}},
            {"transpose": {"axis": [2, 0, 1]}},
            {"div": {"x": 255.0}},
            {"expandDim": {"axis": 0}},
        ]

    def postprocessing(self):
        # Note: Temporary workaround for the mismatch in output tensor order between the original ONNX model and DXNN.
        #       The _wrapper function will be removed once the issue is properly fixed.
        # return yolov7_face_postprocessing
        return yolov7_face_postprocessing_wrapper
