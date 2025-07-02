import numpy as np
import onnx
import pytest


@pytest.fixture
def test_onnx():
    nodes = [
        onnx.helper.make_node(
            "Add",
            inputs=["node1_input", "node1_init"],
            outputs=["node1_output"],
            name="node1",
        ),
        onnx.helper.make_node(
            "Mul",
            inputs=["node1_output", "node2_init"],
            outputs=["node2_output"],
            name="node2",
        ),
    ]

    graph = onnx.helper.make_graph(
        nodes=nodes,
        name="test_graph",
        inputs=[onnx.helper.make_tensor_value_info("node1_input", onnx.TensorProto.FLOAT, [1, 3, 10, 10])],
        outputs=[onnx.helper.make_empty_tensor_value_info("node2_output")],
        initializer=[
            onnx.numpy_helper.from_array(np.array([1], dtype=np.float32), name="node1_init"),
            onnx.numpy_helper.from_array(np.array([2], dtype=np.float32), name="node2_init"),
        ],
    )
    model = onnx.helper.make_model(graph)
    return model


@pytest.fixture
def test_onnx_path(tmp_path, test_onnx):
    onnx_path = tmp_path / "test.onnx"
    onnx.save(test_onnx, onnx_path)
    return onnx_path
