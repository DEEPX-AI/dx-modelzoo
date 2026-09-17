import numpy as np
import torch

from dx_modelzoo.dataset import DATASET_REGISTRY
from dx_modelzoo.evaluator import EVALUATOR_REGISTRY
from dx_modelzoo.evaluator.embedding_cache import load_or_build_embedding, resolve_embedding_cache_path
from dx_modelzoo.evaluator.image_retrieval_evaluator import (
    _load_or_build_image_embeddings,
    _project_text_embeddings,
    _write_embedding_batches,
)


class IdentityModel:
    def ln_final(self, values):
        return values

    text_projection = torch.eye(2)


def test_projects_eot_token_into_clip_text_embedding():
    transformer_output = np.array(
        [
            [[1.0, 0.0], [0.0, 2.0]],
            [[0.0, 3.0], [4.0, 0.0]],
        ],
        dtype=np.float32,
    )
    token_ids = np.array([[1, 99], [99, 1]], dtype=np.int64)

    result = _project_text_embeddings(transformer_output, token_ids, IdentityModel())

    np.testing.assert_allclose(result, [[0.0, 2.0], [0.0, 3.0]])


def test_builds_image_embeddings_when_cache_is_missing(tmp_path):
    embeddings, labels = _load_or_build_image_embeddings(
        tmp_path / "images.npy",
        tmp_path / "labels.npy",
        lambda: (np.ones((2, 4), dtype=np.float32), np.array([0, 1], dtype=np.int64)),
    )

    np.testing.assert_array_equal(embeddings, np.ones((2, 4), dtype=np.float32))
    np.testing.assert_array_equal(labels, [0, 1])


def test_image_retrieval_components_are_registered():
    assert EVALUATOR_REGISTRY.get("image_retrieval").__name__ == "ImageRetrievalEvaluator"
    assert DATASET_REGISTRY.get("ImageNetText").__name__ == "ImageNetText"


def test_writes_embedding_batches_directly_to_npy(tmp_path):
    embedding_path = tmp_path / "image_embeddings.npy"
    label_path = tmp_path / "image_labels.npy"
    batches = [
        (np.array([[1.0, 2.0]], dtype=np.float32), np.array([3], dtype=np.int64)),
        (np.array([[4.0, 5.0], [6.0, 7.0]], dtype=np.float32), np.array([8, 9], dtype=np.int64)),
    ]

    embeddings, labels = _write_embedding_batches(batches, 3, embedding_path, label_path)

    np.testing.assert_array_equal(embeddings, [[1.0, 2.0], [4.0, 5.0], [6.0, 7.0]])
    np.testing.assert_array_equal(labels, [3, 8, 9])
    np.testing.assert_array_equal(np.load(embedding_path), embeddings)
    np.testing.assert_array_equal(np.load(label_path), labels)


def test_discovers_and_persists_embedding_cache_beside_model(tmp_path):
    model_path = tmp_path / "clip-text.onnx"
    model_path.touch()
    expected = np.array([[1.0, 2.0]], dtype=np.float32)
    calls = 0

    def build():
        nonlocal calls
        calls += 1
        return expected

    first = load_or_build_embedding(None, str(model_path), "text_embedding.npy", build)
    second = load_or_build_embedding(None, str(model_path), "text_embedding.npy", lambda: np.zeros((1, 2)))

    np.testing.assert_array_equal(first, expected)
    np.testing.assert_array_equal(second, expected)
    assert calls == 1
    assert (tmp_path / "text_embedding.npy").is_file()


def test_defaults_image_embedding_caches_beside_model(tmp_path):
    model_path = tmp_path / "clip-text.onnx"
    model_path.touch()

    embedding_path = resolve_embedding_cache_path(None, str(model_path), "imagenet_image_embeddings.npy")
    label_path = resolve_embedding_cache_path(None, str(model_path), "imagenet_image_labels.npy")

    assert embedding_path == tmp_path / "imagenet_image_embeddings.npy"
    assert label_path == tmp_path / "imagenet_image_labels.npy"
