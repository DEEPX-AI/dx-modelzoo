"""Tests for dx_modelzoo.postprocessing.decode_utils."""

import numpy as np

from dx_modelzoo.postprocessing.decode_utils import (
    apply_obj_cls_score,
    build_nms_input,
    build_yolox_grids,
    cxcywh_to_xyxy,
    generate_grid_center_priors,
    infer_input_size,
    sigmoid,
    split_box_cls,
    split_box_cls_xyxy,
    transpose_output,
    xyxy_to_cxcywh,
)


class TestSigmoid:
    def test_zero(self):
        result = sigmoid(np.array([0.0]))
        np.testing.assert_almost_equal(result, [0.5])

    def test_large_positive(self):
        result = sigmoid(np.array([100.0]))
        np.testing.assert_almost_equal(result, [1.0])

    def test_large_negative(self):
        result = sigmoid(np.array([-100.0]))
        np.testing.assert_almost_equal(result, [0.0])

    def test_batch(self):
        result = sigmoid(np.array([-1.0, 0.0, 1.0]))
        assert result.shape == (3,)
        assert result[0] < 0.5
        assert result[2] > 0.5


class TestCxcywhToXyxy:
    def test_basic(self):
        boxes = np.array([[10, 10, 4, 6]], dtype=np.float32)
        result = cxcywh_to_xyxy(boxes)
        np.testing.assert_array_almost_equal(result, [[8, 7, 12, 13]])

    def test_batch(self):
        boxes = np.array([[5, 5, 2, 2], [10, 10, 4, 4]], dtype=np.float32)
        result = cxcywh_to_xyxy(boxes)
        assert result.shape == (2, 4)
        np.testing.assert_array_almost_equal(result[0], [4, 4, 6, 6])


class TestXyxyToCxcywh:
    def test_basic(self):
        boxes = np.array([[4, 4, 6, 6]], dtype=np.float32)
        result = xyxy_to_cxcywh(boxes)
        np.testing.assert_array_almost_equal(result, [[5, 5, 2, 2]])

    def test_roundtrip(self):
        original = np.array([[10, 20, 30, 40]], dtype=np.float32)
        cxcywh = xyxy_to_cxcywh(original)
        restored = cxcywh_to_xyxy(cxcywh)
        np.testing.assert_array_almost_equal(restored, original)


class TestApplyObjClsScore:
    def test_basic(self):
        # [N, 5+C]: cx, cy, w, h, obj_conf, cls0, cls1
        output = np.array([[10, 10, 4, 4, 0.9, 0.8, 0.2]], dtype=np.float32)
        boxes, scores, class_ids = apply_obj_cls_score(output)
        assert boxes.shape == (1, 4)
        assert scores.shape == (1,)
        assert class_ids[0] == 0  # cls0 > cls1


class TestSplitBoxCls:
    def test_basic(self):
        output = np.array([[10, 10, 4, 4, 0.9, 0.1]], dtype=np.float32)
        boxes, scores, class_ids = split_box_cls(output)
        assert boxes.shape == (1, 4)
        np.testing.assert_almost_equal(scores[0], 0.9)
        assert class_ids[0] == 0


class TestSplitBoxClsXyxy:
    def test_basic(self):
        output = np.array([[0, 0, 10, 10, 0.3, 0.7]], dtype=np.float32)
        boxes, scores, class_ids = split_box_cls_xyxy(output)
        np.testing.assert_array_equal(boxes[0], [0, 0, 10, 10])
        assert class_ids[0] == 1


class TestTransposeOutput:
    def test_transposes_when_c_lt_n(self):
        output = np.zeros((1, 5, 100), dtype=np.float32)
        result = transpose_output(output)
        assert result.shape == (1, 100, 5)

    def test_no_transpose_when_n_lt_c(self):
        output = np.zeros((1, 100, 5), dtype=np.float32)
        result = transpose_output(output)
        assert result.shape == (1, 100, 5)


class TestBuildNmsInput:
    def test_basic(self):
        boxes = np.zeros((5, 4))
        scores = np.zeros(5)
        class_ids = np.zeros(5)
        result = build_nms_input(boxes, scores, class_ids)
        assert "boxes" in result
        assert "scores" in result
        assert "class_ids" in result

    def test_with_extra(self):
        result = build_nms_input(np.zeros((3, 4)), np.zeros(3), np.zeros(3), extra=np.zeros((3, 2)))
        assert "extra" in result


class TestBuildYoloxGrids:
    def test_basic(self):
        grids, strides = build_yolox_grids(640, [8, 16, 32])
        assert grids.shape[0] == 1
        assert strides.shape[0] == 1
        expected_points = (640//8)**2 + (640//16)**2 + (640//32)**2
        assert grids.shape[1] == expected_points


class TestGenerateGridCenterPriors:
    def test_basic(self):
        priors = generate_grid_center_priors(64, 64, [8, 16])
        # (64/8)^2 + (64/16)^2 = 64 + 16 = 80
        assert priors.shape == (80, 3)


class TestInferInputSize:
    def test_basic(self):
        # 640x640 with strides [8,16,32] → (80^2+40^2+20^2) = 8400 points
        size = infer_input_size(8400, [8, 16, 32])
        assert size == 640
