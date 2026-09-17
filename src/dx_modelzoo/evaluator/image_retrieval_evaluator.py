from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import numpy as np
from loguru import logger

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.evaluator import EVALUATOR_REGISTRY, EvaluatorBase
from dx_modelzoo.evaluator.embedding_cache import load_or_build_embedding, resolve_embedding_cache_path
from dx_modelzoo.session import SessionBase

_IMAGE_EMBEDDING_CACHE = "imagenet_image_embeddings.npy"
_IMAGE_LABEL_CACHE = "imagenet_image_labels.npy"
_TEXT_EMBEDDING_CACHE = "imagenet_text_embedding.npy"
_IMAGE_EMBEDDING_BATCH_SIZE = 32


def _load_or_build_image_embeddings(
    embedding_path: Optional[Path],
    label_path: Optional[Path],
    builder: Callable[[], Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    if embedding_path and label_path and embedding_path.is_file() and label_path.is_file():
        return np.load(embedding_path).astype(np.float32), np.load(label_path).astype(np.int64)
    return builder()


def _write_embedding_batches(
    batches: Iterable[Tuple[np.ndarray, np.ndarray]],
    total_items: int,
    embedding_path: Path,
    label_path: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    labels = np.lib.format.open_memmap(label_path, mode="w+", dtype=np.int64, shape=(total_items,))
    embeddings = None
    offset = 0

    for batch_embeddings, batch_labels in batches:
        batch_embeddings = np.asarray(batch_embeddings, dtype=np.float32)
        batch_labels = np.asarray(batch_labels, dtype=np.int64).reshape(-1)
        if embeddings is None:
            embeddings = np.lib.format.open_memmap(
                embedding_path,
                mode="w+",
                dtype=np.float32,
                shape=(total_items, batch_embeddings.shape[1]),
            )
        end = offset + len(batch_labels)
        embeddings[offset:end] = batch_embeddings
        labels[offset:end] = batch_labels
        offset = end

    if embeddings is None or offset != total_items:
        raise ValueError(f"Expected {total_items} embedding items, wrote {offset}")
    embeddings.flush()
    labels.flush()
    return embeddings, labels


def _project_text_embeddings(transformer_output: np.ndarray, token_ids: np.ndarray, model) -> np.ndarray:
    import torch

    with torch.no_grad():
        output = model.ln_final(torch.from_numpy(transformer_output))
        eot_indices = torch.from_numpy(token_ids).argmax(dim=-1)
        output = output[torch.arange(output.shape[0]), eot_indices]
        if model.text_projection is not None:
            output = output @ model.text_projection
    return output.numpy()


@EVALUATOR_REGISTRY.register("image_retrieval")
class ImageRetrievalEvaluator(EvaluatorBase):
    """Evaluate text-to-image retrieval against precomputed ImageNet embeddings."""

    def __init__(
        self,
        session: SessionBase,
        dataset: DatasetBase,
        clip_model_name: str = "RN50-quickgelu",
        clip_pretrained: str = "openai",
        image_embedding_path: Optional[str] = None,
        image_label_path: Optional[str] = None,
        image_dataset_path: Optional[str] = None,
        text_embedding_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(session, dataset, workers=1, **kwargs)
        self.clip_model_name = clip_model_name
        self.clip_pretrained = clip_pretrained
        self.image_embedding_path = image_embedding_path
        self.image_label_path = image_label_path
        self.image_dataset_path = image_dataset_path
        self.text_embedding_path = text_embedding_path
        self.reference_model = None

    def init_metrics(self) -> Dict[str, float]:
        import open_clip
        import torch

        session_path = getattr(self.session, "path", "")
        image_embedding_path = resolve_embedding_cache_path(
            self.image_embedding_path,
            session_path,
            _IMAGE_EMBEDDING_CACHE,
        )
        image_label_path = resolve_embedding_cache_path(
            self.image_label_path,
            session_path,
            _IMAGE_LABEL_CACHE,
        )
        self.image_embedding_path = str(image_embedding_path) if image_embedding_path else None
        self.image_label_path = str(image_label_path) if image_label_path else None

        logger.info(
            "Loading CLIP text reference '{}' (pretrained='{}')",
            self.clip_model_name,
            self.clip_pretrained,
        )
        self.reference_model, _, self.image_preprocess = open_clip.create_model_and_transforms(
            self.clip_model_name,
            pretrained=self.clip_pretrained,
            device="cpu",
        )
        self.reference_model.eval()
        self.text_transformer_outputs = {}
        self.dataset_token_ids = np.asarray(self.dataset.token_ids, dtype=np.int64)

        def build_reference_text_embedding() -> np.ndarray:
            with torch.no_grad():
                reference_transformer = self.reference_model.transformer(
                    torch.from_numpy(self.dataset.prompts),
                    attn_mask=self.reference_model.attn_mask,
                ).numpy()
            # Store [embedding_dim, num_classes], matching zero-shot caches.
            return _project_text_embeddings(
                reference_transformer,
                self.dataset_token_ids,
                self.reference_model,
            ).T

        cached_reference = load_or_build_embedding(
            self.text_embedding_path,
            session_path,
            _TEXT_EMBEDDING_CACHE,
            build_reference_text_embedding,
        )
        if cached_reference.ndim != 2 or cached_reference.shape[1] != len(self.dataset_token_ids):
            raise ValueError(f"Unexpected text embedding shape: {cached_reference.shape}")
        self.reference_text_embeddings = cached_reference.T
        self.image_embeddings, self.image_labels = _load_or_build_image_embeddings(
            Path(self.image_embedding_path) if self.image_embedding_path else None,
            Path(self.image_label_path) if self.image_label_path else None,
            self._build_image_embeddings,
        )
        if self.image_embeddings.ndim != 2 or self.image_embeddings.shape[1] != self.reference_text_embeddings.shape[1]:
            raise ValueError(f"Unexpected image embedding shape: {self.image_embeddings.shape}")
        if self.image_labels.shape != (len(self.image_embeddings),):
            raise ValueError(f"Unexpected image label shape: {self.image_labels.shape}")
        return {"current_count": 0}

    def _build_image_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        import torch
        from PIL import Image
        from torch.utils.data import DataLoader, Dataset

        root = Path(
            self.image_dataset_path or Path(os.environ.get("DATA_ROOT", "download/datasets")) / "ILSVRC2012" / "val"
        )
        classes = sorted(path for path in root.iterdir() if path.is_dir())
        items = [(path, label) for label, class_dir in enumerate(classes) for path in sorted(class_dir.glob("*.JPEG"))]
        if not items:
            raise FileNotFoundError(f"No ImageNet JPEG files found under {root}")

        preprocess = self.image_preprocess

        class ImageNetImages(Dataset):
            def __len__(self):
                return len(items)

            def __getitem__(self, idx):
                path, label = items[idx]
                with Image.open(path) as image:
                    return preprocess(image.convert("RGB")), label

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reference_model.to(device)
        self.reference_model.eval()
        loader = DataLoader(
            ImageNetImages(),
            batch_size=_IMAGE_EMBEDDING_BATCH_SIZE,
            shuffle=False,
            num_workers=min(8, os.cpu_count() or 1),
            pin_memory=device.type == "cuda",
        )

        def encoded_batches():
            for images, batch_labels in loader:
                with torch.inference_mode():
                    batch_embeddings = (
                        self.reference_model.encode_image(images.to(device, non_blocking=True)).float().cpu().numpy()
                    )
                yield batch_embeddings, batch_labels.numpy()

        try:
            if self.image_embedding_path and self.image_label_path:
                return _write_embedding_batches(
                    encoded_batches(),
                    len(items),
                    Path(self.image_embedding_path),
                    Path(self.image_label_path),
                )

            embeddings, labels = zip(*encoded_batches())
            image_embeddings = np.concatenate(embeddings, axis=0).astype(np.float32)
            image_labels = np.concatenate(labels, axis=0).astype(np.int64)
            return image_embeddings, image_labels
        finally:
            self.reference_model.to("cpu")
            if device.type == "cuda":
                torch.cuda.empty_cache()

    def extract_inputs(self, batch_data: Any) -> Any:
        inputs, _ = batch_data
        if isinstance(inputs, np.ndarray) and inputs.ndim == 2:
            return np.expand_dims(inputs, axis=0)
        return inputs

    def process_batch_result(self, batch_data: Any, output: Any, metrics_state: Dict[str, float]) -> Dict[str, float]:
        _, index = batch_data
        actual = output[0] if isinstance(output, (list, tuple)) else output
        actual = np.asarray(actual, dtype=np.float32)
        self.text_transformer_outputs[int(np.asarray(index).reshape(-1)[0])] = actual[0]
        metrics_state["current_count"] += actual.shape[0]
        return metrics_state

    def compute_final_metrics(self, metrics_state: Dict[str, float]) -> dict:
        count = metrics_state["current_count"]
        actual_transformer = np.stack(
            [self.text_transformer_outputs[index] for index in range(len(self.dataset_token_ids))]
        )
        actual_text = _project_text_embeddings(actual_transformer, self.dataset_token_ids, self.reference_model)
        actual_text_norm = actual_text / np.maximum(np.linalg.norm(actual_text, axis=1, keepdims=True), 1e-12)
        reference_text_norm = self.reference_text_embeddings / np.maximum(
            np.linalg.norm(self.reference_text_embeddings, axis=1, keepdims=True), 1e-12
        )
        text_cosine = float(np.mean(np.sum(actual_text_norm * reference_text_norm, axis=1)))
        text_max_abs = float(np.max(np.abs(actual_text - self.reference_text_embeddings)))

        image_norm = self.image_embeddings / np.maximum(
            np.linalg.norm(self.image_embeddings, axis=1, keepdims=True), 1e-12
        )
        logits = image_norm @ actual_text_norm.T
        ranking = np.argsort(-logits, axis=1)
        top1 = float(np.mean(ranking[:, 0] == self.image_labels) * 100)
        top5 = float(np.mean(np.any(ranking[:, :5] == self.image_labels[:, None], axis=1)) * 100)
        fps = count / self.total_inference_time if self.total_inference_time > 0 else 0.0
        return self._finalize(
            metric_names=["Top1 Accuracy", "Top5 Accuracy", "Text Cosine Similarity", "Text Max Abs Error"],
            metric_values=[top1, top5, text_cosine, text_max_abs],
            fps=fps,
        )

    def format_progress_desc(self, metrics_state: Dict[str, float], current_fps: float) -> str:
        return f"Image retrieval | samples:{metrics_state['current_count']} FPS:{current_fps:.1f}"
