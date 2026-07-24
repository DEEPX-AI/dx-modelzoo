"""Tests for dx_modelzoo.postprocessing.nms."""

import numpy as np

from dx_modelzoo.postprocessing.nms import NMS, nms_numpy


class TestNmsNumpy:
    def test_basic_nms(self):
        boxes = np.array([
            [0, 0, 10, 10],
            [1, 1, 11, 11],  # high IoU with first
            [50, 50, 60, 60],  # separate box
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        keep = nms_numpy(boxes, scores, iou_threshold=0.5)
        assert 0 in keep
        assert 2 in keep
        assert 1 not in keep

    def test_empty_input(self):
        boxes = np.zeros((0, 4), dtype=np.float32)
        scores = np.zeros(0, dtype=np.float32)
        keep = nms_numpy(boxes, scores, iou_threshold=0.5)
        assert len(keep) == 0

    def test_single_box(self):
        boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        keep = nms_numpy(boxes, scores, iou_threshold=0.5)
        assert list(keep) == [0]

    def test_high_threshold_keeps_all(self):
        boxes = np.array([
            [0, 0, 10, 10],
            [1, 1, 11, 11],
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        keep = nms_numpy(boxes, scores, iou_threshold=0.99)
        assert len(keep) == 2


class TestNMSPostprocessor:
    def test_dict_input(self):
        nms = NMS(conf_thres=0.3, iou_thres=0.5)
        inputs = {
            "boxes": np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32),
            "scores": np.array([0.9, 0.8, 0.7], dtype=np.float32),
            "class_ids": np.array([0, 0, 1], dtype=np.float64),
        }
        result = nms(inputs)
        assert isinstance(result, np.ndarray)
        assert result.shape[1] == 6
        assert result.shape[0] <= 3

    def test_ndarray_input(self):
        nms = NMS(conf_thres=0.3, iou_thres=0.5)
        # [M, 6] format: x1, y1, x2, y2, score, class_id
        detections = np.array([
            [0, 0, 10, 10, 0.9, 0],
            [1, 1, 11, 11, 0.8, 0],
            [50, 50, 60, 60, 0.7, 1],
        ], dtype=np.float32)
        result = nms(detections)
        assert result.shape[1] == 6

    def test_below_threshold_filtered(self):
        nms = NMS(conf_thres=0.5, iou_thres=0.5)
        inputs = {
            "boxes": np.array([[0, 0, 10, 10]], dtype=np.float32),
            "scores": np.array([0.1], dtype=np.float32),  # Below threshold
            "class_ids": np.array([0], dtype=np.float64),
        }
        result = nms(inputs)
        assert result.shape[0] == 0

    def test_with_coord_scaling(self):
        nms = NMS(conf_thres=0.1, iou_thres=0.5, pad_resize=True)
        inputs = {
            "boxes": np.array([[10, 10, 100, 100]], dtype=np.float32),
            "scores": np.array([0.9], dtype=np.float32),
            "class_ids": np.array([0], dtype=np.float64),
        }
        result = nms(inputs, input_hw=(640, 640), origin_hw=(480, 640))
        assert result.shape[0] == 1
        assert result.shape[1] == 6
