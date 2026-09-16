import numpy as np

from dx_modelzoo.postprocessing.rtdetr import decode_rows, unpad_and_scale


def test_decode_rows_respects_count_and_reorders_columns():
    # ONNX emits [class, score, x1, y1, x2, y2] rows padded past ``count``.
    detections = np.array(
        [
            [3.0, 0.9, 0.0, 0.0, 320.0, 640.0],
            [7.0, 0.5, 320.0, 0.0, 640.0, 320.0],
            [1.0, 0.1, 0.0, 0.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    out = decode_rows((detections, np.array([2])), input_hw=(640, 640), origin_hw=(320, 640), pad_location="direct")

    assert out.shape == (2, 6)
    # boxes scaled from the 640x640 model grid back to the 640x320 original
    np.testing.assert_allclose(out[0, :4], [0.0, 0.0, 320.0, 320.0])
    np.testing.assert_allclose(out[:, 4], [0.9, 0.5])
    np.testing.assert_allclose(out[:, 5], [3.0, 7.0])


def test_unpad_and_scale_direct_clips_to_origin():
    boxes = np.array([[-10.0, -10.0, 700.0, 700.0]])
    out = unpad_and_scale(boxes, input_hw=(640, 640), origin_hw=(100, 200), pad_location="direct")
    np.testing.assert_allclose(out, [[0.0, 0.0, 200.0, 100.0]])
